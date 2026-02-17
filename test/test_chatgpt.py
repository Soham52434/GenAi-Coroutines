
import unittest
import os
import json
import asyncio
from genai_coroutines import ResponsesRequest, ResponsesProcessor, parse_responses

# Ensure we have an API key
API_KEY = os.getenv("OPENAI_API_KEY")

class TestChatGPT(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if not API_KEY:
            self.skipTest("OPENAI_API_KEY not found in environment")
        
        # ResponsesProcessor currently takes NO arguments
        self.processor = ResponsesProcessor()

    async def test_basic_chat(self):
        """Test basic chat completion"""
        prompts = ["Hello, introduce yourself briefly."]
        request = ResponsesRequest(
            api_key=API_KEY,      # api_key must be passed here
            model="gpt-3.5-turbo",
            system_prompt="You are a helpful assistant.",
            user_prompts=prompts,
            max_tokens=50
        )
        
        results = await self.processor.process_batch(request)
        self.assertTrue(results["success"])
        self.assertEqual(len(results["results"]), 1)
        
        result = results["results"][0]
        self.assertTrue(result["success"])
        
        # Verify usage bug fix: should be a dict, not a string
        if "usage" in result:
            self.assertIsInstance(result["usage"], dict, "usage should be a dict (Bug fix verified)")
            self.assertIn("total_tokens", result["usage"])
        
        # Verify parsing helper
        parsed = parse_responses(results)
        self.assertEqual(len(parsed), 1)
        self.assertIsInstance(parsed[0], str)
        self.assertTrue(len(parsed[0]) > 0)

    async def test_batch_processing(self):
        """Test batch processing with multiple prompts"""
        prompts = ["Count to 3", "Say 'hello'", "What is 2+2?"]
        request = ResponsesRequest(
            api_key=API_KEY,
            model="gpt-3.5-turbo",
            user_prompts=prompts,
            max_tokens=20
        )
        
        results = await self.processor.process_batch(request)
        self.assertTrue(results["success"])
        self.assertEqual(len(results["results"]), 3)
        
        # Verify 1:1 mapping in parsing helper
        parsed = parse_responses(results)
        self.assertEqual(len(parsed), 3, "parse_responses() should return 1:1 mapping")
        for i, text in enumerate(parsed):
            self.assertTrue(len(text) > 0, f"Response {i} was empty")

    async def test_structured_output(self):
        """Test structured output with json_object"""
        prompts = ["Generate a JSON object with keys 'name' and 'age' for a fictional character."]
        request = ResponsesRequest(
            api_key=API_KEY,
            model="gpt-3.5-turbo-1106", # Supports JSON mode
            user_prompts=prompts,
            response_format={"type": "json_object"},
            max_tokens=100
        )
        
        results = await self.processor.process_batch(request)
        result = results["results"][0]
        self.assertTrue(result["success"])
        
        parsed = parse_responses(results)
        text = parsed[0]
        
        # Verify it's valid JSON
        try:
            data = json.loads(text)
            self.assertIn("name", data)
            self.assertIn("age", data)
        except json.JSONDecodeError:
            self.fail(f"Response was not valid JSON: {text}")

    async def test_error_handling(self):
        """Test with invalid API key"""
        processor = ResponsesProcessor()
        request = ResponsesRequest(
            api_key="invalid_key", # Invalid key here
            model="gpt-3.5-turbo",
            user_prompts=["This should fail"],
            max_tokens=10
        )
        results = await processor.process_batch(request)
        
        # Check overall success might be true if partial failures allowed, check individual result
        result = results["results"][0]
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    async def test_parse_responses_1_to_1(self):
        """Verify parse_responses handles mixed success/failure and maintains order"""
        prompts = ["A", "B", "C", "D", "E"]
        request = ResponsesRequest(
            api_key=API_KEY,
            model="gpt-3.5-turbo",
            user_prompts=prompts,
            max_tokens=5
        )
        
        results = await self.processor.process_batch(request)
        parsed = parse_responses(results)
        
        self.assertEqual(len(parsed), len(prompts), "Parsed list length MUST match prompt length")


class TestChatGPTLogic(unittest.TestCase):
    def test_parse_standard_message(self):
        """Verify parsing of standard message content"""
        mock_results = {
            "results": [
                {
                    "success": True,
                    "raw_response": json.dumps({
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "Hello"}]
                            }
                        ]
                    })
                }
            ]
        }
        parsed = parse_responses(mock_results)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0], "Hello")

    def test_parse_tool_calls(self):
        """Verify handling of tool-call-only responses (returns raw JSON)"""
        raw_json = json.dumps({
            "output": [
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": "{}"
                }
            ]
        })
        mock_results = {
            "results": [
                {
                    "success": True,
                    "raw_response": raw_json
                }
            ]
        }
        parsed = parse_responses(mock_results)
        self.assertEqual(len(parsed), 1)
        # Should return the full raw JSON string
        self.assertEqual(parsed[0], raw_json)

    def test_parse_mixed_batch_order(self):
        """Verify 1:1 mapping and order preservation in batch"""
        # Batch of 3: success, failure, tool-call
        fail_result = {"success": False, "error": "Rate limit"}
        
        success_json = json.dumps({
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Success"}]}]
        })
        success_result = {"success": True, "raw_response": success_json}
        
        tool_json = json.dumps({"output": [{"type": "function_call"}]})
        tool_result = {"success": True, "raw_response": tool_json}

        mock_results = {
            "results": [success_result, fail_result, tool_result]
        }
        
        parsed = parse_responses(mock_results)
        self.assertEqual(len(parsed), 3, "Must return exactly 3 items")
        self.assertEqual(parsed[0], "Success")
        self.assertEqual(parsed[1], "") # Failed item returns empty string
        self.assertEqual(parsed[2], tool_json) # Tool call returns raw JSON

if __name__ == "__main__":
    unittest.main()
