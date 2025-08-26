#!/usr/bin/env python3
"""
Simple HTTP server to serve the test page
"""

import http.server
import socketserver
import os

# Set the port
PORT = 8080

# Change to the directory with the HTML file
os.chdir('/home/beats/Desktop/ai-music-generation')

# Create a handler
Handler = http.server.SimpleHTTPRequestHandler

# Create a server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    print(f"Open http://localhost:{PORT}/test_page_direct.html in your browser")
    # Serve until interrupted
    httpd.serve_forever()
