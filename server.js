import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

// Load environment variables for the main server
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load root server .env first (PORT, ALLOWED_ORIGINS, etc.)
dotenv.config({ path: path.join(__dirname, ".env") });

// Load Corporate AI .env BEFORE importing rag.js
// (ESM static imports are hoisted, so we must load env vars here
//  and use a dynamic import below to avoid the hoisting issue)
const ragEnvPath = path.join(__dirname, "MANO-Corporate-Website-AI", ".env");
dotenv.config({ path: ragEnvPath, override: true });

const app = express();
const PORT = process.env.PORT || 5555;

app.use(express.json());

const allowedOrigins = (process.env.ALLOWED_ORIGINS || "http://localhost:5173")
    .split(",")
    .map((o) => o.trim());

app.use(
    cors({
        origin: allowedOrigins,
        methods: ["GET", "POST"],
        credentials: true,
    })
);

// Dynamic import AFTER env vars are loaded (avoids ESM hoisting issue with dotenv v17)
const { answerQuestion, getVectorCount } = await import("./MANO-Corporate-Website-AI/rag_engine/rag.js");

// ─── Routes: Corporate Website AI ─────────────────────────────────────────────

app.get("/chat-corporate/health", async (req, res) => {
    try {
        const count = await getVectorCount();
        res.json({
            status: "ok",
            service: "MANO Corporate AI API",
            vectors_indexed: count,
        });
    } catch (err) {
        res.status(500).json({ error: "Health check failed", detail: err.message });
    }
});

app.post("/chat-corporate/chat", async (req, res) => {
    const { question } = req.body;

    if (!question || !question.trim()) {
        return res.status(400).json({ error: "Question cannot be empty." });
    }

    try {
        const result = await answerQuestion(question.trim());
        res.json({ answer: result.answer, sources: result.sources });
    } catch (err) {
        console.error("[CHAT CORP ERROR]", err);
        res.status(500).json({ error: "Internal server error.", detail: err.message });
    }
});

// ─── Main Server Start ────────────────────────────────────────────────────────

app.listen(PORT, () => {
    console.log(`\n[CENTRALISED AI SERVER] Running at http://localhost:${PORT}`);
    console.log(`  -> Corporate AI Chat:    POST http://localhost:${PORT}/chat-corporate/chat`);
    console.log(`  -> Corporate AI Health:  GET  http://localhost:${PORT}/chat-corporate/health`);
});
