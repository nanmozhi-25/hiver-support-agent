"""
Vercel Serverless Function Handler for Hiver AI Support Agent API.
Handles POST /api/process, GET /api/metrics, and GET /api/golden on Vercel deployment.
"""

import os
import sys
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler

# Add root directory to sys.path so src module can be imported on Vercel
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.pipeline import SupportAgentPipeline

pipeline_instance = None


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == "/api/metrics":
            metrics_path = os.path.join(os.path.dirname(__file__), "..", "results", "metrics.json")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(json.dumps({"error": "Metrics not found"}).encode("utf-8"))
            return

        elif path == "/api/golden":
            golden_path = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "golden_set.csv")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if os.path.exists(golden_path):
                import pandas as pd
                df = pd.read_csv(golden_path)
                samples = df.to_dict(orient="records")
                self.wfile.write(json.dumps(samples).encode("utf-8"))
            else:
                self.wfile.write(json.dumps([]).encode("utf-8"))
            return

        else:
            self.send_response(404)
            self.end_headers()

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
