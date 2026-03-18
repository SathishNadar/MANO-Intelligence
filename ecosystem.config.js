export default {
  apps: [
    // 1. Current Centralized AI Server (Node.js)
    {
      name: "mano-ai-central-server",
      script: "./server.js",
      instances: 1, // Single instance to prevent chromadb concurrency issues
      exec_mode: "fork",
      env: {
        NODE_ENV: "production",
      }
    },
    
    // 2. Future Agent Example (Python) 
    // You can just uncomment and modify this when you create a new Python agent
    /*
    {
      name: "future-rag-backend",
      script: "venv/bin/python",
      args: "main.py",
      cwd: "./Future-Agent-Folder", // The folder where your future agent lives
      interpreter: "none",          // Treat script as literal path
      env: {
        PORT: 5556
      }
    }
    */
  ]
};
