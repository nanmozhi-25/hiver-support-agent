"""
Self-Contained Local Web Server & Interactive AI Support Agent Dashboard.
Serves interactive Web UI at http://localhost:8000 with REST API endpoints:
- POST /api/process : Process query through pipeline
- GET  /api/metrics : Get evaluation metrics JSON
- GET  /api/golden  : Get Golden evaluation set samples
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

from src.pipeline import SupportAgentPipeline

PORT = 8000
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "metrics.json")
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "golden_set.csv")

# Global pipeline instance
pipeline_instance = None


class SupportAgentHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(json.dumps({"error": "Metrics not found. Run evaluation first."}).encode("utf-8"))
            return

        elif path == "/api/golden":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if os.path.exists(GOLDEN_PATH):
                import pandas as pd
                df = pd.read_csv(GOLDEN_PATH)
                samples = df.to_dict(orient="records")
                self.wfile.write(json.dumps(samples).encode("utf-8"))
            else:
                self.wfile.write(json.dumps([]).encode("utf-8"))
            return

        elif path == "/" or path == "/index.html":
            self.path = "/index.html"
            return SimpleHTTPRequestHandler.do_GET(self)
            
        else:
            return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == "/api/process":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode("utf-8"))
                text = data.get("text", "").strip()
                
                if not text:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Query text cannot be empty"}).encode("utf-8"))
                    return

                global pipeline_instance
                if pipeline_instance is None:
                    pipeline_instance = SupportAgentPipeline()

                result = pipeline_instance.process(text)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port: int = PORT):
    global pipeline_instance
    print("Initializing AI Support Agent Pipeline...")
    pipeline_instance = SupportAgentPipeline()

    os.chdir(STATIC_DIR)
    server_address = ("", port)
    httpd = HTTPServer(server_address, SupportAgentHandler)
    
    print(f"\n=========================================================================")
    print(f"HIVER AI SUPPORT AGENT WEB SERVER IS LIVE!")
    print(f"Local Web App Host Link: http://localhost:{port}")
    print(f"=========================================================================\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server...")
        httpd.server_close()


if __name__ == "__main__":
    os.makedirs(STATIC_DIR, exist_ok=True)
    run_server()
