
import unittest
import os
import json
import asyncio
from genai_coroutines import DocumentConfig, DocumentProcessor, parse_documents

# Ensure we have an API key
API_KEY = os.getenv("DATALAB_API_KEY")

class TestChandraOCR(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if not API_KEY:
            self.skipTest("DATALAB_API_KEY not found in environment")
        
        # Processor takes a DocumentConfig object, which takes the API key
        self.config = DocumentConfig(api_key=API_KEY)
        self.processor = DocumentProcessor(self.config)
        self.image_path = "test/images/ext3.png"
        
        # Verify test image exists
        if not os.path.exists(self.image_path):
            self.fail(f"Test image not found: {self.image_path}")

        # Read image bytes
        with open(self.image_path, "rb") as f:
            self.image_bytes = f.read()

    async def test_basic_ocr_json(self):
        """Test basic OCR with default JSON output"""
        # Create a new config for this test
        config = DocumentConfig(
            api_key=API_KEY,
            api_url="https://www.datalab.to/api/v1/marker",
            output_format="json"
        )
        # Processor can't change config dynamically, need new processor
        processor = DocumentProcessor(config)
        
        results = await processor.process_multiparts([self.image_bytes])
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertTrue(result["success"], f"OCR failed: {result.get('error')}")
        self.assertIn("json_response", result)
        
        # Verify cost_breakdown bug fix: should be a dict, not a string
        if "cost_breakdown" in result:
            self.assertIsInstance(result["cost_breakdown"], dict, "cost_breakdown should be a dict (Bug 1 fix)")
            self.assertIn("input_cost_cents", result["cost_breakdown"])
        
        # Verify parsing helper
        parsed = parse_documents(results)
        self.assertEqual(len(parsed), 1)
        self.assertIsInstance(parsed[0], str)
        self.assertTrue(len(parsed[0]) > 0, "Parsed output should not be empty")

    async def test_output_formats(self):
        """Test html and markdown output formats"""
        formats = ["html", "markdown"]
        
        for fmt in formats:
            with self.subTest(format=fmt):
                config = DocumentConfig(api_key=API_KEY, output_format=fmt)
                processor = DocumentProcessor(config)
                
                results = await processor.process_multiparts([self.image_bytes])
                result = results[0]
                self.assertTrue(result["success"])
                
                # Check parsed output matches expected format content
                parsed = parse_documents(results)
                content = parsed[0]
                
                if fmt == "html":
                    self.assertIn("<", content)  # Basic HTML check
                elif fmt == "markdown":
                    self.assertTrue(len(content) > 0)

    async def test_structured_extraction(self):
        """Test structured data extraction with page_schema"""
        schema = {
            "type": "object",
            "properties": {
                "document_type": {"type": "string"},
                "confidence_score": {"type": "number"}
            }
        }
        
        config = DocumentConfig(
            api_key=API_KEY,
            page_schema=json.dumps(schema)
        )
        processor = DocumentProcessor(config)
        
        results = await processor.process_multiparts([self.image_bytes])
        result = results[0]
        self.assertTrue(result["success"])
        
        # Parse documents should return the full JSON string for structured output
        parsed = parse_documents(results)
        parsed_json = json.loads(parsed[0])
        
        # The result structure might vary based on API, but we check if we got JSON back
        self.assertIsInstance(parsed_json, (dict, list))
        
    async def test_error_handling(self):
        """Test with invalid API key"""
        config = DocumentConfig(api_key="invalid_key")
        processor = DocumentProcessor(config)
        results = await processor.process_multiparts([self.image_bytes])
        
        result = results[0]
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIsInstance(result["error"], str)

class TestChandraParsing(unittest.TestCase):
    def test_parse_json_format(self):
        """Verify parsing of standard JSON output with children"""
        mock_result = {
            "success": True,
            "json_response": json.dumps({
                "json": {
                    "children": [
                        {"html": "<p>Page 1</p>"},
                        {"html": "<p>Page 2</p>"}
                    ]
                }
            })
        }
        parsed = parse_documents([mock_result])
        self.assertEqual(parsed[0], "<p>Page 1</p>\n<p>Page 2</p>")

    def test_parse_html_format(self):
        """Verify parsing of direct HTML output"""
        mock_result = {
            "success": True,
            "json_response": json.dumps({
                "html": "<div>Full Doc</div>"
            })
        }
        parsed = parse_documents([mock_result])
        self.assertEqual(parsed[0], "<div>Full Doc</div>")

    def test_parse_markdown_format(self):
        """Verify parsing of markdown output"""
        mock_result = {
            "success": True,
            "json_response": json.dumps({
                "markdown": "# Title\n\nBody"
            })
        }
        parsed = parse_documents([mock_result])
        self.assertEqual(parsed[0], "# Title\n\nBody")

    def test_parse_paginated(self):
        """Verify parsing of paginated output"""
        mock_result = {
            "success": True,
            "json_response": json.dumps({
                "pages": [
                    {"html": "<p>P1</p>"},
                    {"markdown": "P2"}
                ]
            })
        }
        parsed = parse_documents([mock_result])
        self.assertEqual(parsed[0], "<p>P1</p>\nP2")

    def test_parse_structured(self):
        """Verify parsing of structured output (page_schema)"""
        # Should return full JSON
        raw_json = json.dumps({"page_schema": True, "data": {"name": "Test"}})
        mock_result = {
            "success": True,
            "json_response": raw_json
        }
        parsed = parse_documents([mock_result])
        self.assertEqual(parsed[0], raw_json)

if __name__ == "__main__":
    unittest.main()
