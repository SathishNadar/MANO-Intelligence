// rag.js — Core RAG logic: ChromaDB + Embeddings + Groq LLM
import { ChromaClient } from "chromadb";
import Groq from "groq-sdk";
import { pipeline } from "@xenova/transformers";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs/promises";
import dotenv from "dotenv";

// Load .env from the rag_engine folder (where GROQ_API_KEY lives)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// override:true ensures this wins even if root server.js loaded .env first
dotenv.config({ path: path.join(__dirname, "..", ".env"), override: true });

// ─── Config ───────────────────────────────────────────────────────────────────
const CHROMA_PATH = process.env.CHROMA_PATH || path.join(__dirname, "chroma_data");
const CHUNKS_FILE = process.env.KNOWLEDGE_CHUNKS_PATH || path.join(__dirname, "..", "knowledge_base", "chunks.json");
const COLLECTION = "mano_kb";
const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2"; // same model used during indexing

function readEnvNumber(name, fallback) {
    const raw = process.env[name];
    if (raw === undefined || raw === null || raw === "") return fallback;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

const DEFAULT_TOP_K = Math.round(clamp(readEnvNumber("RAG_DEFAULT_TOP_K", 24), 1, 100));
const MAX_RETRIEVAL_K = Math.round(clamp(readEnvNumber("RAG_MAX_RETRIEVAL_K", 40), 5, 200));
const MIN_LOCATION_RETRIEVAL_K = Math.round(clamp(readEnvNumber("RAG_MIN_LOCATION_RETRIEVAL_K", 30), 1, 200));
const MIN_GENERAL_RETRIEVAL_K = Math.round(clamp(readEnvNumber("RAG_MIN_GENERAL_RETRIEVAL_K", 32), 1, 200));
const LOCATION_DISTANCE_THRESHOLD = clamp(readEnvNumber("RAG_LOCATION_DISTANCE_THRESHOLD", 1.8), 0.1, 5);
const GENERAL_DISTANCE_THRESHOLD = clamp(readEnvNumber("RAG_GENERAL_DISTANCE_THRESHOLD", 1.9), 0.1, 5);
const LOCATION_RESULT_LIMIT = Math.round(clamp(readEnvNumber("RAG_LOCATION_RESULT_LIMIT", 8), 1, 30));
const LOCATION_HINT_RETRIEVAL_K = Math.round(clamp(readEnvNumber("RAG_LOCATION_HINT_RETRIEVAL_K", 25), 1, 100));
const RAG_DEBUG = String(process.env.RAG_DEBUG || "false").toLowerCase() === "true";
const MAX_LLM_CONTEXT_CHARS = Math.round(clamp(readEnvNumber("RAG_MAX_LLM_CONTEXT_CHARS", 7000), 1500, 20000));
const MAX_LLM_OUTPUT_TOKENS = Math.round(clamp(readEnvNumber("RAG_MAX_LLM_OUTPUT_TOKENS", 450), 100, 1500));
const HYBRID_MIN_SEMANTIC_SCORE = clamp(readEnvNumber("RAG_HYBRID_MIN_SEMANTIC_SCORE", 0.28), 0, 1);
const HYBRID_MIN_KEYWORD_SCORE = clamp(readEnvNumber("RAG_HYBRID_MIN_KEYWORD_SCORE", 0.12), 0, 1);

const rawSemanticWeight = clamp(readEnvNumber("RAG_HYBRID_SEMANTIC_WEIGHT", 0.7), 0, 1);
const rawKeywordWeight = clamp(readEnvNumber("RAG_HYBRID_KEYWORD_WEIGHT", 0.3), 0, 1);
const weightSum = rawSemanticWeight + rawKeywordWeight;
const HYBRID_SEMANTIC_WEIGHT = weightSum > 0 ? rawSemanticWeight / weightSum : 0.7;
const HYBRID_KEYWORD_WEIGHT = weightSum > 0 ? rawKeywordWeight / weightSum : 0.3;

const STOPWORDS = new Set([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "i", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "please", "the", "their", "them", "to", "us", "was", "we", "what",
    "when", "where", "which", "who", "why", "with", "you", "your", "about", "details", "tell", "give", "explain"
]);

// ─── Lazy-loaded singletons ───────────────────────────────────────────────────
let _embedder = null;
let _collection = null;
let _groq = null;
let _localChunkStore = null;

async function getEmbedder() {
    if (!_embedder) {
        console.log("[RAG] Loading embedding model (all-MiniLM-L6-v2)...");
        _embedder = await pipeline("feature-extraction", EMBED_MODEL);
        console.log("[RAG] Embedding model ready.");
    }
    return _embedder;
}

async function getCollection() {
    if (!_collection) {
        const chromaUrl = process.env.CHROMA_URL || "http://localhost:8000";
        console.log(`[RAG] Connecting to ChromaDB at: ${chromaUrl}`);
        const client = new ChromaClient({ path: chromaUrl });
        _collection = await client.getOrCreateCollection({ name: COLLECTION });
        const count = await _collection.count();
        console.log(`[RAG] Collection '${COLLECTION}' has ${count} vectors`);
    }
    return _collection;
}

function getGroq() {
    // Read from process.env every call so dotenv order doesn't matter
    const key = process.env.GROQ_API_KEY || "";
    const model = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
    if (!_groq && key) {
        _groq = {
            client: new Groq({ apiKey: key }),
            model,
        };
        console.log(`[RAG] Groq model loaded: ${model}`);
    }
    return _groq;
}

async function getLocalChunkStore() {
    if (_localChunkStore) return _localChunkStore;

    try {
        const raw = await fs.readFile(CHUNKS_FILE, "utf-8");
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) {
            _localChunkStore = { chunks: [], embeddings: [], loaded: false };
            return _localChunkStore;
        }

        const chunks = parsed
            .filter((item) => item && typeof item.content === "string" && item.content.trim())
            .map((item, idx) => ({
                id: item.id || `local_chunk_${idx}`,
                content: item.content,
                page: item.page_name || "Unknown",
                url: item.url || "",
                source_file: item.source_file || "",
                section: item.section_num ?? idx,
            }));

        _localChunkStore = { chunks, embeddings: [], loaded: true };
        console.log(`[RAG] Local KB loaded from chunks.json: ${chunks.length} chunks`);
        return _localChunkStore;
    } catch (err) {
        console.warn(`[RAG] Local chunks fallback unavailable at ${CHUNKS_FILE}: ${err.message}`);
        _localChunkStore = { chunks: [], embeddings: [], loaded: false };
        return _localChunkStore;
    }
}

function cosineSimilarity(a = [], b = []) {
    if (!a.length || !b.length || a.length !== b.length) return 0;

    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i += 1) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function ensureLocalEmbeddings(store) {
    if (!store.loaded || !store.chunks.length) return;
    if (store.embeddings.length === store.chunks.length) return;

    console.log("[RAG] Building local fallback embeddings from chunks.json...");
    const embeddings = [];
    for (const chunk of store.chunks) {
        const emb = await embedText(chunk.content);
        embeddings.push(emb);
    }

    store.embeddings = embeddings;
    console.log(`[RAG] Local fallback embeddings ready: ${store.embeddings.length}`);
}

async function queryLocalChunks(question, nResults) {
    const store = await getLocalChunkStore();
    if (!store.loaded || !store.chunks.length) return [];

    await ensureLocalEmbeddings(store);
    const qEmb = await embedText(question);

    const ranked = store.chunks
        .map((chunk, idx) => {
            const sim = cosineSimilarity(qEmb, store.embeddings[idx] || []);
            return {
                doc: chunk.content,
                meta: {
                    page: chunk.page,
                    url: chunk.url,
                    section: chunk.section,
                    source_file: chunk.source_file,
                },
                dist: Math.max(0, 1 - sim),
            };
        })
        .sort((a, b) => a.dist - b.dist)
        .slice(0, Math.max(1, nResults));

    return ranked;
}

// ─── Embed a text string → float array ───────────────────────────────────────
async function embedText(text) {
    const embedder = await getEmbedder();
    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
}

// ─── System Prompt ────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are an official, professional assistant for Mano Projects Pvt. Ltd.,
a Construction & Project Management firm based in Mumbai, India, established in 2010.

RULES:
- Answer ONLY using the context provided below.
- Provide highly PRECISE and DETAILED answers. Include exact figures, names, years, and locations whenever available in context.
- Prioritize factual correctness over verbosity; do not add assumptions.
- If multiple relevant points exist, structure the answer clearly with concise bullet points.
- Do NOT repeat the same point in different wording.
- Keep each bullet unique and non-overlapping.
- Avoid restating the same service/process item more than once.
- If asked for process/workflow, return step-by-step sequence from context.
- If asked about a location or address, provide the FULL exact location from the context (e.g., plot numbers, street names, full address).
- If asked about services or features, list them comprehensively with details.
- Always answer in full, professional sentences or well-formatted paragraphs/bullet points.
- Do NOT make up any information not in the context.
- If the answer is not in the context, say exactly:
  "I don't have that specific information. Please contact us directly at our office."
- Never discuss competitors or topics unrelated to MANO Projects.`;

const LOCATION_QUERY_REGEX = /\b(where|location|located|address|head office|office address|contact\s+address)\b/i;
const MISSION_VISION_QUERY_REGEX = /\b(mission|vision|vission)\b/i;
const ESTABLISHED_QUERY_REGEX = /\b(when.*establish|when.*founded|when.*started|established|founded|started)\b/i;
const MISSION_VISION_HINT_RETRIEVAL_K = Math.round(clamp(readEnvNumber("RAG_MISSION_VISION_HINT_RETRIEVAL_K", 20), 5, 80));

function isLocationQuestion(question) {
    return LOCATION_QUERY_REGEX.test(question || "");
}

function isMissionVisionQuestion(question) {
    return MISSION_VISION_QUERY_REGEX.test(question || "");
}

function isEstablishedQuestion(question) {
    return ESTABLISHED_QUERY_REGEX.test((question || "").toLowerCase());
}

function addressSignalScore(doc = "", meta = {}) {
    const text = `${doc}\n${meta?.page || ""}\n${meta?.source_file || ""}`.toLowerCase();
    let score = 0;

    if (text.includes("head office address")) score += 8;
    if (text.includes("contact information")) score += 4;
    if (text.includes("address")) score += 3;
    if (text.includes("dadar")) score += 3;
    if (text.includes("l.n road") || text.includes("ln road")) score += 3;
    if (text.includes("mumbai - 400014") || text.includes("400014")) score += 4;

    return score;
}

function missionVisionSignalScore(doc = "", meta = {}) {
    const text = `${doc}\n${meta?.page || ""}\n${meta?.source_file || ""}`.toLowerCase();
    let score = 0;

    if (text.includes("vision")) score += 4;
    if (text.includes("mission")) score += 4;
    if (text.includes("about us") || text.includes("about_the_company") || text.includes("about the company")) score += 3;
    if ((meta?.source_file || "").toLowerCase().includes("about_us")) score += 4;
    if (text.includes("values") || text.includes("ethics") || text.includes("stakeholder")) score += 2;

    return score;
}

function extractMissionVision(documents = []) {
    for (const doc of documents) {
        const lines = (doc || "")
            .replace(/\r/g, "")
            .split("\n")
            .map((line) => line.trim())
            .filter(Boolean);

        if (!lines.length) continue;

        const visionIdx = lines.findIndex((line) => /^vision$/i.test(line));
        const missionIdx = lines.findIndex((line) => /^mission$/i.test(line));

        if (visionIdx === -1 && missionIdx === -1) continue;

        let visionLines = [];
        let missionLines = [];

        if (visionIdx !== -1) {
            const visionEnd = missionIdx !== -1 && missionIdx > visionIdx ? missionIdx : Math.min(visionIdx + 7, lines.length);
            visionLines = lines
                .slice(visionIdx + 1, visionEnd)
                .filter((line) => !/^about the team$/i.test(line));
        }

        if (missionIdx !== -1) {
            missionLines = lines
                .slice(missionIdx + 1)
                .filter((line) => !/^about the team$/i.test(line) && !/^contact information$/i.test(line))
                .slice(0, 8);
        }

        if (visionLines.length || missionLines.length) {
            return {
                vision: visionLines,
                mission: missionLines,
            };
        }
    }

    return null;
}

function extractExactAddress(documents = []) {
    for (const doc of documents) {
        const normalized = (doc || "").replace(/\r/g, "");

        const blockMatch = normalized.match(/Head Office Address:\s*\n?([^\n]+(?:\n(?!Phone:|Email:).+)*)/i);
        if (blockMatch?.[1]) {
            return blockMatch[1].split("\n").map((line) => line.trim()).filter(Boolean).join(", ");
        }

        const inlineMatch = normalized.match(/Head Office Address:\s*([^\n]+)/i);
        if (inlineMatch?.[1]) {
            return inlineMatch[1].trim();
        }
    }

    return "";
}

function uniqueByTextRanked(items = []) {
    const seen = new Set();
    const out = [];
    for (const item of items) {
        const key = (item?.doc || "").trim();
        if (!key || seen.has(key)) continue;
        seen.add(key);
        out.push(item);
    }
    return out;
}

function dedupeAnswerLines(text = "") {
    const lines = text.split("\n");
    const seen = new Set();
    const out = [];

    for (const line of lines) {
        const normalized = line
            .toLowerCase()
            .replace(/^\s*[-*\d.)\s]+/, "")
            .replace(/[^a-z0-9\s]/g, "")
            .replace(/\s+/g, " ")
            .trim();

        if (!normalized) {
            if (out.length > 0 && out[out.length - 1] !== "") out.push("");
            continue;
        }

        if (normalized.length >= 20 && seen.has(normalized)) {
            continue;
        }

        seen.add(normalized);
        out.push(line);
    }

    return out.join("\n").trim();
}

function buildTrimmedContext(relevant = []) {
    let used = 0;
    const chunks = [];

    for (const item of relevant) {
        const doc = (item?.doc || "").trim();
        if (!doc) continue;
        if (used >= MAX_LLM_CONTEXT_CHARS) break;

        const remaining = MAX_LLM_CONTEXT_CHARS - used;
        const clipped = doc.length > remaining ? doc.slice(0, remaining) : doc;
        chunks.push(clipped);
        used += clipped.length;
    }

    return chunks.join("\n\n---\n\n");
}

function buildExtractiveFallbackAnswer(question, relevant = [], retryAfterSeconds = null) {
    const qTokens = tokenize(question);
    const candidates = [];

    for (const item of relevant) {
        const lines = (item?.doc || "")
            .replace(/\r/g, "")
            .split("\n")
            .map((line) => line.trim())
            .filter(Boolean)
            .filter((line) => line.length >= 20);

        for (const line of lines) {
            candidates.push({
                line,
                score: keywordOverlapScore(qTokens, line),
                dist: item?.dist ?? 2,
            });
        }
    }

    candidates.sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        return a.dist - b.dist;
    });

    const picked = [];
    const seen = new Set();
    for (const c of candidates) {
        const key = c.line.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        picked.push(c.line);
        if (picked.length >= 4) break;
    }

    if (!picked.length) {
        return "I don't have that specific information. Please contact us directly at our office.";
    }

    const prefix = retryAfterSeconds
        ? `The AI model is currently rate-limited. Please retry in about ${Math.ceil(retryAfterSeconds / 60)} minute(s). Meanwhile, based on our knowledge base:`
        : "Based on our knowledge base:";

    return `${prefix}\n${picked.map((line) => `- ${line}`).join("\n")}`;
}

function extractEstablishedYearFromDocs(documents = []) {
    for (const doc of documents) {
        const text = (doc || "").replace(/\r/g, " ");
        const match = text.match(/established\s+in\s+([A-Za-z]+\s+)?(\d{4})/i);
        if (match?.[2]) {
            return match[2];
        }
    }
    return null;
}

function tokenize(text = "") {
    return (text || "")
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, " ")
        .split(/\s+/)
        .filter((token) => token.length > 1 && !STOPWORDS.has(token));
}

function semanticScoreFromDistance(distance) {
    const d = typeof distance === "number" ? distance : 3;
    return 1 / (1 + Math.max(0, d));
}

function keywordOverlapScore(queryTokens, doc = "") {
    const qTokens = Array.from(new Set(queryTokens));
    if (qTokens.length === 0) return 0;

    const docTokens = new Set(tokenize(doc));
    if (docTokens.size === 0) return 0;

    let overlap = 0;
    for (const token of qTokens) {
        if (docTokens.has(token)) overlap += 1;
    }

    return overlap / qTokens.length;
}

function hybridRankCandidates(candidates, question, limit) {
    const queryTokens = tokenize(question);

    const ranked = candidates
        .map((candidate) => {
            const semantic = semanticScoreFromDistance(candidate.dist);
            const lexical = keywordOverlapScore(queryTokens, candidate.doc);
            const hybrid = (semantic * HYBRID_SEMANTIC_WEIGHT) + (lexical * HYBRID_KEYWORD_WEIGHT);
            return { ...candidate, semantic, lexical, hybrid };
        })
        .sort((a, b) => {
            if (b.hybrid !== a.hybrid) return b.hybrid - a.hybrid;
            return a.dist - b.dist;
        })
        .filter(({ semantic, lexical }) => semantic > HYBRID_MIN_SEMANTIC_SCORE || lexical > HYBRID_MIN_KEYWORD_SCORE);

    return uniqueByTextRanked(ranked).slice(0, limit);
}

// ─── Main RAG: embed question → retrieve chunks → Gemini answer ───────────────
export async function answerQuestion(question, topK = DEFAULT_TOP_K) {
    let collection = null;
    let useLocalFallback = false;

    try {
        collection = await getCollection();
        const count = await collection.count();
        if (count === 0) {
            console.warn("[RAG] Chroma collection is empty; switching to local chunks.json fallback.");
            useLocalFallback = true;
        }
    } catch (err) {
        console.warn(`[RAG] Chroma unavailable (${err.message}); switching to local chunks.json fallback.`);
        useLocalFallback = true;
    }

    // 1. Embed the question
    const queryEmbedding = await embedText(question);

    const locationQuestion = isLocationQuestion(question);
    const missionVisionQuestion = isMissionVisionQuestion(question);
    const retrievalK = locationQuestion
        ? Math.min(Math.max(topK, MIN_LOCATION_RETRIEVAL_K), MAX_RETRIEVAL_K)
        : Math.min(Math.max(topK, MIN_GENERAL_RETRIEVAL_K), MAX_RETRIEVAL_K);

    let docs = [];
    let metas = [];
    let distances = [];

    // 2. Retrieve top-K relevant chunks
    if (useLocalFallback) {
        const localResults = await queryLocalChunks(question, retrievalK);
        docs = localResults.map((r) => r.doc);
        metas = localResults.map((r) => r.meta);
        distances = localResults.map((r) => r.dist);
    } else {
        const results = await collection.query({
            queryEmbeddings: [queryEmbedding],
            nResults: retrievalK,
            include: ["documents", "metadatas", "distances"],
        });

        docs = results.documents[0] || [];
        metas = results.metadatas[0] || [];
        distances = results.distances[0] || [];
    }

    if (!docs.length) {
        return {
            answer: "The knowledge base is empty. Please run the indexing step first.",
            sources: [],
        };
    }

    // 3. Filter low-relevance results (distance > 1.5 = very dissimilar)
    if (RAG_DEBUG) {
        console.log(`\n[DEBUG] Question: ${question}`);
        docs.forEach((d, i) => {
            console.log(`[DEBUG] Dist: ${distances[i].toFixed(2)} | Source: ${metas[i].source_file} | Preview: ${d.substring(0, 50).replace(/\n/g, " ")}`);
        });
    }

    const candidates = docs
        .map((doc, i) => ({ doc, meta: metas[i], dist: distances[i] }))
        .filter(({ doc }) => Boolean(doc));

    let relevant;
    if (locationQuestion) {
        relevant = candidates
            .map((c) => ({ ...c, signal: addressSignalScore(c.doc, c.meta) }))
            .sort((a, b) => {
                if (b.signal !== a.signal) return b.signal - a.signal;
                return a.dist - b.dist;
            })
            .filter(({ dist, signal }) => signal > 0 || dist < LOCATION_DISTANCE_THRESHOLD)
            .slice(0, LOCATION_RESULT_LIMIT);
    } else {
        relevant = hybridRankCandidates(
            candidates.filter(({ dist }) => dist < GENERAL_DISTANCE_THRESHOLD),
            question,
            topK
        );
    }

    if (relevant.length === 0) {
        return {
            answer: "I don't have that specific information. Please contact us directly at our office.",
            sources: [],
        };
    }

    const context = relevant.map(({ doc }) => doc).join("\n\n---\n\n");
    const sources = [...new Set(relevant.map(({ meta }) => meta.page))];

    if (isEstablishedQuestion(question)) {
        const year = extractEstablishedYearFromDocs(relevant.map(({ doc }) => doc));
        if (year) {
            return {
                answer: `MANO Projects Pvt. Ltd. was established in ${year}.`,
                sources,
            };
        }
    }

    if (locationQuestion) {
        const exactAddress = extractExactAddress(relevant.map(({ doc }) => doc));
        if (exactAddress) {
            return {
                answer: `MANO Projects Pvt. Ltd. Head Office Address: ${exactAddress}`,
                sources,
            };
        }

        const locationHint = `${question} head office address contact information dadar l.n road mumbai 400014`;
        let hintedDocs = [];
        let hintedMetas = [];

        if (useLocalFallback || !collection) {
            const hintedLocalResults = await queryLocalChunks(locationHint, LOCATION_HINT_RETRIEVAL_K);
            hintedDocs = hintedLocalResults.map((item) => item.doc);
            hintedMetas = hintedLocalResults.map((item) => item.meta);
        } else {
            const hintedEmbedding = await embedText(locationHint);
            const hintedResults = await collection.query({
                queryEmbeddings: [hintedEmbedding],
                nResults: LOCATION_HINT_RETRIEVAL_K,
                include: ["documents", "metadatas", "distances"],
            });

            hintedDocs = hintedResults.documents?.[0] || [];
            hintedMetas = hintedResults.metadatas?.[0] || [];
        }

        const hintedAddress = extractExactAddress(hintedDocs);
        if (hintedAddress) {
            const hintedSources = [...new Set(hintedMetas.map((meta) => meta?.page).filter(Boolean))];
            return {
                answer: `MANO Projects Pvt. Ltd. Head Office Address: ${hintedAddress}`,
                sources: hintedSources.length ? hintedSources : sources,
            };
        }
    }

    if (missionVisionQuestion) {
        const localStore = await getLocalChunkStore();
        const directMissionDocs = (localStore?.chunks || [])
            .filter((chunk) => /mission|vision|vission/i.test(chunk.content || ""))
            .sort((a, b) => {
                const aAbout = /about_us/i.test(a.source_file || "") ? 1 : 0;
                const bAbout = /about_us/i.test(b.source_file || "") ? 1 : 0;
                return bAbout - aAbout;
            });

        const directMissionExtract = extractMissionVision(directMissionDocs.map((item) => item.content));
        if (directMissionExtract) {
            const lines = [];
            if (directMissionExtract.vision.length) {
                lines.push("MANO Vision:");
                for (const line of directMissionExtract.vision) lines.push(`- ${line}`);
            }
            if (directMissionExtract.mission.length) {
                if (lines.length) lines.push("");
                lines.push("MANO Mission:");
                for (const line of directMissionExtract.mission) lines.push(`- ${line}`);
            }

            return {
                answer: lines.join("\n"),
                sources: ["ABOUT US"],
            };
        }

        const missionVisionHint = `${question} about us mission vision values ethics stakeholder value`;
        let hintedDocs = [];
        let hintedMetas = [];
        let hintedDistances = [];

        if (useLocalFallback || !collection) {
            const hintedLocalResults = await queryLocalChunks(missionVisionHint, MISSION_VISION_HINT_RETRIEVAL_K);
            hintedDocs = hintedLocalResults.map((item) => item.doc);
            hintedMetas = hintedLocalResults.map((item) => item.meta);
            hintedDistances = hintedLocalResults.map((item) => item.dist);
        } else {
            const hintedEmbedding = await embedText(missionVisionHint);
            const hintedResults = await collection.query({
                queryEmbeddings: [hintedEmbedding],
                nResults: MISSION_VISION_HINT_RETRIEVAL_K,
                include: ["documents", "metadatas", "distances"],
            });

            hintedDocs = hintedResults.documents?.[0] || [];
            hintedMetas = hintedResults.metadatas?.[0] || [];
            hintedDistances = hintedResults.distances?.[0] || [];
        }

        const missionCandidates = hintedDocs
            .map((doc, i) => ({ doc, meta: hintedMetas[i], dist: hintedDistances[i] ?? 1.5 }))
            .filter(({ doc }) => Boolean(doc))
            .map((item) => ({ ...item, signal: missionVisionSignalScore(item.doc, item.meta) }))
            .sort((a, b) => {
                if (b.signal !== a.signal) return b.signal - a.signal;
                return a.dist - b.dist;
            });

        const missionExtract = extractMissionVision(missionCandidates.map((item) => item.doc));
        if (missionExtract) {
            const sourceSet = missionCandidates
                .map((item) => item.meta?.page)
                .filter(Boolean);

            const lines = [];
            if (missionExtract.vision.length) {
                lines.push("MANO Vision:");
                for (const line of missionExtract.vision) lines.push(`- ${line}`);
            }
            if (missionExtract.mission.length) {
                if (lines.length) lines.push("");
                lines.push("MANO Mission:");
                for (const line of missionExtract.mission) lines.push(`- ${line}`);
            }

            return {
                answer: lines.join("\n"),
                sources: [...new Set(sourceSet)].slice(0, 6),
            };
        }
    }

    // 4. Call Groq
    const groq = getGroq();
    if (!groq) {
        return {
            answer: "[LLM not configured — set GROQ_API_KEY in .env]",
            sources,
        };
    }

    const fullPrompt = `CONTEXT:\n${buildTrimmedContext(relevant)}\n\nQUESTION: ${question}`;
    try {
        const result = await groq.client.chat.completions.create({
            messages: [
                { role: "system", content: SYSTEM_PROMPT },
                { role: "user", content: fullPrompt }
            ],
            model: groq.model,
            temperature: 0.2, // low temp for factual RAG answers
            max_tokens: MAX_LLM_OUTPUT_TOKENS,
        });
        const rawAnswer = result.choices[0]?.message?.content || "No response generated.";
        const answer = dedupeAnswerLines(rawAnswer);
        return { answer: answer.trim(), sources };
    } catch (err) {
        const isRateLimited = Number(err?.status) === 429 || err?.error?.error?.code === "rate_limit_exceeded";
        const retryAfter = Number(err?.headers?.["retry-after"] || 0);

        if (isRateLimited) {
            const answer = buildExtractiveFallbackAnswer(question, relevant, retryAfter > 0 ? retryAfter : null);
            return { answer, sources };
        }

        console.error("[RAG] LLM call failed:", err?.message || err);
        const answer = buildExtractiveFallbackAnswer(question, relevant, null);
        return { answer, sources };
    }
}

// ─── Health: return vector count ──────────────────────────────────────────────
export async function getVectorCount() {
    try {
        const collection = await getCollection();
        return await collection.count();
    } catch {
        try {
            const store = await getLocalChunkStore();
            return store.chunks.length;
        } catch {
            return 0;
        }
    }
}
