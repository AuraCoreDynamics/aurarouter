"""
Example: Grid Consumer App

This example demonstrates how to interact with AuraRouter
when running as a service on AuraGrid.
"""

import asyncio
import json
from typing import Optional


class SimpleGridConsumer:
    """
    Example grid application that uses AuraRouter services.
    
    Demonstrates both synchronous (gRPC) and asynchronous (events) patterns.
    """

    def __init__(self, app_name: str = "GridConsumer"):
        """Initialize the consumer app."""
        self.app_name = app_name
        self.service_client = None
        self.event_publisher = None
        self.event_consumer = None

    async def initialize(self):
        """
        Initialize connections to grid services.
        
        In a real app, AuraGrid would provide these automatically
        when running on the grid.
        """
        # In actual AuraGrid environment, these would be provided by:
        # from auragrid import create_service_client, get_event_publisher
        
        # For this example, we'll stub them
        print(f"[{self.app_name}] Initializing connections to AuraGrid services...")
        # self.service_client = create_service_client("UnifiedRouterService")
        # self.event_publisher = get_event_publisher()
        # self.event_consumer = get_event_consumer()

    async def example_sync_rpc_call(self) -> Optional[str]:
        """
        Example 1: Call AuraRouter via synchronous gRPC.
        
        Best for: Interactive queries, immediate feedback needed
        """
        print("\n=== Example 1: Synchronous RPC Call ===")
        
        task = "Create a Python class that implements a LRU cache"
        language = "python"
        
        print(f"Task: {task}")
        print(f"Language: {language}")
        
        # In real AuraGrid environment:
        # result = await self.service_client.intelligent_code_gen(
        #     task=task,
        #     language=language
        # )
        # print(f"Result: {result['result']}")
        # return result['result']
        
        # For demo:
        print("(In real grid environment, would call UnifiedRouterService.intelligent_code_gen)")
        return None

    async def example_async_events(self):
        """
        Example 2: Publish task to event topic; subscribe to results.
        
        Best for: Bulk operations, non-blocking processing
        """
        print("\n=== Example 2: Asynchronous Event-Based Call ===")
        
        task = "Large code generation for REST API"
        language = "typescript"
        
        # Create routing request
        from aurarouter.auragrid.events import EventBridge
        
        bridge = EventBridge()
        request = bridge.create_routing_request(
            task=task,
            language=language,
            context={"module": "my_api_module"}
        )
        
        print(f"Request ID: {request['request_id']}")
        print(f"Task: {task}")
        print(f"Return topic: {request['return_topic']}")
        
        # In real AuraGrid environment:
        # await self.event_publisher.publish(
        #     topic="aurarouter.routing_requests",
        #     payload=json.dumps(request).encode()
        # )
        #
        # # Subscribe to results
        # async for result_event in self.event_consumer.consume(request['return_topic']):
        #     result_data = json.loads(result_event.payload)
        #     print(f"Result: {result_data['result']}")
        
        print("(In real grid environment, would publish event and wait for results)")

    async def example_individual_services(self):
        """
        Example 3: Call individual services separately.
        
        Best for: Multi-step workflows with intermediate processing
        """
        print("\n=== Example 3: Individual Service Calls ===")
        
        task = "Extract data from CSV and create a report"
        
        # Step 1: Classify intent
        print(f"\nStep 1: Classify intent")
        print(f"Task: {task}")
        # In real environment:
        # classification = await self.router_client.classify_intent(task)
        # print(f"Classification: {classification['classification']}")
        
        # Step 2: Generate plan
        print(f"\nStep 2: Generate plan")
        # In real environment:
        # plan = await self.reasoning_client.generate_plan(task)
        # for i, step in enumerate(plan['steps'], 1):
        #     print(f"  {i}. {step}")
        
        # Step 3: Generate code for each step
        print(f"\nStep 3: Generate code")
        steps = [
            "Read CSV file and parse data",
            "Filter data by date range",
            "Generate summary statistics",
            "Create formatted report"
        ]
        
        for step in steps:
            print(f"  Generating: {step}")
            # In real environment:
            # code = await self.coding_client.generate_code(
            #     plan_step=step,
            #     language="python"
            # )
            # print(f"  Generated code: {code['code'][:100]}...")

    async def example_error_handling(self):
        """
        Example 4: Proper error handling and resilience.
        """
        print("\n=== Example 4: Error Handling ===")
        
        task = "This is a valid task"
        
        try:
            # Simulate service call with error handling
            print(f"Calling service with task: {task}")
            
            # In real environment:
            # try:
            #     result = await self.service_client.intelligent_code_gen(
            #         task=task,
            #         language="python"
            #     )
            #     if result['success']:
            #         return result['result']
            #     else:
            #         print(f"Service error: {result.get('error', 'Unknown error')}")
            # except TimeoutError:
            #     print("Service call timed out; try async events instead")
            # except Exception as e:
            #     print(f"Service unavailable: {e}")
            
            print("(Error handling patterns demonstrated)")
            
        except Exception as e:
            print(f"Caught error: {e}")

    async def run_all_examples(self):
        """Run all example scenarios."""
        print("╔════════════════════════════════════════════════╗")
        print("║  AuraRouter on AuraGrid - Consumer Examples   ║")
        print("╚════════════════════════════════════════════════╝")
        
        await self.initialize()
        
        await self.example_sync_rpc_call()
        await self.example_async_events()
        await self.example_individual_services()
        await self.example_error_handling()
        
        print("\n╔════════════════════════════════════════════════╗")
        print("║  Examples Complete                            ║")
        print("║  (Would write actual code in real AuraGrid)   ║")
        print("╚════════════════════════════════════════════════╝")


async def main():
    """Run example consumer app."""
    consumer = SimpleGridConsumer(app_name="DocumentGenerator")
    await consumer.run_all_examples()


if __name__ == "__main__":
    asyncio.run(main())
