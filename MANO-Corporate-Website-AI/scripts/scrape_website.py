import asyncio
import os
import re
from urllib.parse import urlparse, urljoin
from playwright.async_api import async_playwright

# Update this to your live domain when deployed (e.g., https://www.mano-projects.com)
BASE_URL = "http://localhost:5173"

# Path where raw text files will be stored
OUTPUT_DIR = "knowledge_base/pages"

# Do not crawl these types of assets or external links
IGNORE_EXTENSIONS = [".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".mp4", ".css", ".js", ".ico"]

def is_valid_internal_url(url, base_domain):
    """
    Check if URL belongs to the same domain, isn't an anchor link to the same page,
    and doesn't point to an ignored file extension.
    """
    try:
        parsed = urlparse(url)
        # Handle relative urls
        if not parsed.netloc:
            return True
        # Check domain
        if parsed.netloc != base_domain:
            return False
        # Check extensions
        if any(parsed.path.lower().endswith(ext) for ext in IGNORE_EXTENSIONS):
            return False
        return True
    except Exception:
        return False

def clean_url_for_filename(url):
    """
    Convert a URL path like /services/project-management to a clean filename: services_project_management
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return "landing_page"
    
    clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', path)
    return clean_name

async def extract_links_from_page(page, base_url):
    """
    Extract all hrefs from the current page and resolve them to absolute URLs.
    """
    hrefs = await page.evaluate(r"""() => {
        return Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href'));
    }""")
    
    valid_links = set()
    base_domain = urlparse(base_url).netloc
    
    for href in hrefs:
        if not href or href.startswith("mailto:") or href.startswith("tel:") or href.startswith("#"):
            continue
            
        absolute_url = urljoin(base_url, href)
        
        # Strip trailing slashes and hash fragments to avoid duplicates like /services and /services#top
        parsed_abs = urlparse(absolute_url)
        clean_abs = absolute_url.split('#')[0].rstrip('/')
        
        if is_valid_internal_url(clean_abs, base_domain):
            valid_links.add(clean_abs)
            
    return valid_links

async def scrape_page_content(page, url):
    """
    Navigate to the page, scroll down to trigger lazy loading, and extract text chunks.
    """
    print(f"  Scraping: {url}")
    await page.goto(url, wait_until="networkidle", timeout=45000)
    
    # Scroll to trigger lazy-loaded content
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(1500)
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_timeout(500)

    # Extract block by block to separate sections
    page_blocks = await page.evaluate("""() => {
        // Remove navigation and footer to remove noise across pages
        document.querySelectorAll('script, style, noscript, nav, footer, header').forEach(el => el.remove());

        let results = [];
        let sectionNum = 1;

        // Try to find the main container (usually the parent of <section> tags)
        let sections = document.querySelectorAll('section');
        let mainContainer = null;
        
        if (sections.length > 0) {
            mainContainer = sections[0].parentElement;
        } else {
            // Fallback
            mainContainer = document.querySelector('#root') || document.querySelector('main') || document.body;
        }

        // Iterate over direct children of this main container
        if (mainContainer) {
            Array.from(mainContainer.children).forEach(child => {
                let text = child.innerText;
                if (text) {
                    text = text.trim();
                    if (text.length > 15) {
                        results.push(`============================================================\\n[ SECTION ${sectionNum} ]\\n============================================================\\n` + text);
                        sectionNum++;
                    }
                }
            });
        }
        return results;
    }""")
    
    return page_blocks

async def crawl_and_scrape():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    visited_urls = set()
    urls_to_visit = [BASE_URL.rstrip('/')]
    
    # Optional: if you still want to force certain routes, add them to `urls_to_visit` here
    # urls_to_visit.extend([f"{BASE_URL}/careers", f"{BASE_URL}/services"])

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        print(f"\n[INFO] Starting dynamic crawler at BASE_URL: {BASE_URL}")

        while urls_to_visit:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            visited_urls.add(current_url)
            
            try:
                # 1. Scrape content
                blocks = await scrape_page_content(page, current_url)
                
                # 2. Extract internal links for continued crawling
                new_links = await extract_links_from_page(page, current_url)
                for link in new_links:
                    if link not in visited_urls and link not in urls_to_visit:
                        urls_to_visit.append(link)

                # 3. Save text if content found
                if blocks:
                    label = clean_url_for_filename(current_url)
                    output_text = f"PAGE: {label.upper().replace('_',' ')}\nURL: {current_url}\n\n"
                    
                    for block in blocks:
                        lines = [ln.strip() for ln in block.splitlines()]
                        clean_lines = []
                        for ln in lines:
                            if ln:
                                clean_lines.append(ln)
                            elif len(clean_lines) > 0 and clean_lines[-1] != "":
                                clean_lines.append("") 
                                
                        output_text += "\n".join(clean_lines) + "\n\n"
                    
                    file_path = os.path.join(OUTPUT_DIR, f"{label}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(output_text.strip() + "\n")
                    
                    print(f"  [SUCCESS] Saved: {file_path}")
                else:
                    print(f"  [WARNING] No viable text content found on {current_url}")

            except Exception as e:
                print(f"  [ERROR] Failed to scrape {current_url}: {e}")

        await browser.close()

    print(f"\n[SUCCESS] Done! Crawled {len(visited_urls)} pages.")
    print(f"All extracted texts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(crawl_and_scrape())
