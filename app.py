import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from crewai import Crew, Task, Agent
import requests
from langchain_ibm import WatsonxLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import logging

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment setup (these will now be loaded from your .env file)
watsonx_api_key = os.getenv("WATSONX_APIKEY")

WATSONX_URL = os.getenv("WATSONX_URL")
PROJECT_ID = os.getenv("PROJECT_ID")



serper_api_key = os.getenv("SERPER_API_KEY")
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CustomerProfile:
    """Customer profile data structure"""
    customer_id: str
    name: str
    travel_history: List[Dict]
    preferences: Dict
    dietary_restrictions: List[str]
    budget_range: str
    accessibility_needs: List[str]
    past_bookings: List[Dict]

@dataclass
class TripContext:
    """Current trip context"""
    destination: str
    dates: List[str]
    duration: int
    travelers: int
    trip_type: str

class CustomSearchTool:
    """Custom search tool using Serper API directly"""
    
    def __init__(self):
        self.api_key = os.environ.get("SERPER_API_KEY")
        self.base_url = "https://google.serper.dev/search"
        
        if not self.api_key:
            logger.warning("SERPER_API_KEY not found in environment variables")
    
    def search(self, query: str) -> str:
        """Perform web search using Serper API"""
        if not self.api_key:
            return f"Search unavailable - API key not configured for query: {query}"
            
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': 5
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            results = response.json()
            
            # Format results for the agent
            formatted_results = []
            if 'organic' in results:
                for result in results['organic'][:3]:
                    formatted_results.append(
                        f"Title: {result.get('title', '')}\n"
                        f"Link: {result.get('link', '')}\n"
                        f"Snippet: {result.get('snippet', '')}\n"
                    )
            
            return "\n---\n".join(formatted_results) if formatted_results else "No results found"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Search API request failed: {str(e)}")
            return f"Search temporarily unavailable for query: {query}"
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return f"Search error occurred for query: {query}"

class RAGCustomerDatabase:
    """RAG system for customer profiles and preferences"""
    
    def __init__(self):
        try:
            # Initialize embeddings model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Initialize vector store
            embedding_size = 384
            index = faiss.IndexFlatL2(embedding_size)
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )
            
            self._initialize_sample_data()
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise

    def _initialize_sample_data(self):
        """Initialize with sample customer profiles"""
        sample_customers = [
            {
                "customer_id": "sarah_001",
                "name": "Sarah Johnson",
                "profile": "Art enthusiast who loves museums and galleries. Vegetarian with preference for mid-range accommodations. Dislikes crowded places and prefers cultural experiences over nightlife.",
                "travel_history": ["Paris - loved Louvre and small galleries", "Rome - enjoyed Vatican museums", "Amsterdam - preferred quieter neighborhoods"],
                "preferences": {"accommodation": "mid-range hotels", "food": "vegetarian restaurants", "activities": "art museums, galleries, cultural sites"},
                "dietary_restrictions": ["vegetarian"],
                "budget_range": "mid-range",
                "accessibility_needs": [],
                "past_bookings": ["Museum passes", "guided art tours", "boutique hotels"]
            },
            {
                "customer_id": "mike_002", 
                "name": "Mike Chen",
                "profile": "Adventure traveler who enjoys outdoor activities and local cuisine. Flexible budget with preference for authentic experiences. Loves hiking and trying new foods.",
                "travel_history": ["Thailand - loved street food tours", "Peru - enjoyed Machu Picchu trek", "Japan - preferred local neighborhoods"],
                "preferences": {"accommodation": "hostels or local guesthouses", "food": "street food and local cuisine", "activities": "hiking, food tours, adventure sports"},
                "dietary_restrictions": [],
                "budget_range": "budget-conscious",
                "accessibility_needs": [],
                "past_bookings": ["Adventure tours", "food experiences", "budget accommodations"]
            }
        ]
        
        try:
            for customer in sample_customers:
                profile_text = f"{customer['profile']} Travel history: {', '.join(customer['travel_history'])} Preferences: {customer['preferences']}"
                metadata = {
                    "customer_id": customer["customer_id"],
                    "name": customer["name"],
                    "data": customer
                }
                self.vector_store.add_texts([profile_text], metadatas=[metadata])
            logger.info("Sample customer data loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sample data: {str(e)}")

    def retrieve_customer_profile(self, customer_id: str, query_context: str) -> Dict:
        """Retrieve customer profile"""
        try:
            docs = self.vector_store.similarity_search(query_context, k=10)
            
            # Filter for the specific customer
            for doc in docs:
                if doc.metadata.get("customer_id") == customer_id:
                    return doc.metadata["data"]
                    
            # If no match found, try direct lookup from sample data
            if customer_id == "sarah_001":
                return {
                    "customer_id": "sarah_001",
                    "name": "Sarah Johnson",
                    "travel_history": ["Paris - loved Louvre and small galleries", "Rome - enjoyed Vatican museums", "Amsterdam - preferred quieter neighborhoods"],
                    "preferences": {"accommodation": "mid-range hotels", "food": "vegetarian restaurants", "activities": "art museums, galleries, cultural sites"},
                    "dietary_restrictions": ["vegetarian"],
                    "budget_range": "mid-range",
                    "accessibility_needs": [],
                    "past_bookings": ["Museum passes", "guided art tours", "boutique hotels"]
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving customer profile: {str(e)}")
            return None

class WatsonLLMWrapper:
    """Wrapper for Watson LLM to ensure better compatibility with CrewAI"""
    
    def __init__(self, watson_llm):
        self.watson_llm = watson_llm
    
    def invoke(self, prompt, **kwargs):
        """Enhanced invoke method with better error handling"""
        try:
            # To ensure prompt is a string and not too long
            if isinstance(prompt, list):
                prompt = " ".join(str(p) for p in prompt)
            
            # Truncate if too long (but increase limit)
            if len(prompt) > 5000:
                prompt = prompt[:5000] + "..."
            
            # Make the call with timeout
            result = self.watson_llm.invoke(prompt)
            
            # To ensure it return a string
            if not isinstance(result, str):
                result = str(result)
                
            return result
            
        except Exception as e:
            logger.error(f"Watson LLM invoke error: {str(e)}")
            # Return a fallback response instead of raising an exception
            return f"I apologize, but I'm experiencing technical difficulties. Please try again."
    
    def __getattr__(self, name):
        """Delegate other attributes to the underlying Watson LLM"""
        return getattr(self.watson_llm, name)

class TravelConciergeSystem:
    """Main system orchestrating the multi-agent travel planning"""
    
    def __init__(self):
        # Validate environment variables
        required_env_vars = ["WATSONX_APIKEY"]
        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Initialize IBM Watson LLMs with INCREASED parameters
        self.llm_parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 1000,  # INCREASED from 300 to 1000
            "temperature": 0.1,
            "repetition_penalty": 1.05,
            "stop_sequences": ["Human:", "Assistant:", "Task:", "Agent:"]
        }
        
        try:
            # Initialize Watson LLM
            watson_llm = WatsonxLLM(
                model_id="meta-llama/llama-3-3-70b-instruct",
                url=WATSONX_URL,
                params=self.llm_parameters,
                project_id=PROJECT_ID,
            )
            
            # Wrap it for better compatibility
            self.main_llm = WatsonLLMWrapper(watson_llm)
            
            # Test the LLM
            test_response = self.main_llm.invoke("Respond with: LLM ready")
            logger.info(f"Watson LLM test response: {test_response}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Watson LLM: {str(e)}")
            raise
        
        # Initialize tools and RAG system
        self.search_tool = CustomSearchTool()
        self.rag_system = RAGCustomerDatabase()
        
        # Initialize agents with more robust configurations
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup the three specialized agents with improved configurations"""
        
        # Agent 1: Research Intelligence Agent  
        self.research_agent = Agent(
            llm=self.main_llm,
            role="Travel Research Specialist",
            goal="Provide concise destination insights and practical information",
            backstory="You are an expert travel researcher who analyzes destinations and provides clear, actionable insights. You focus on weather, attractions, and practical tips.",
            verbose=False,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=60
        )
        
        # Agent 2: Personalization Agent
        self.personalization_agent = Agent(
            llm=self.main_llm,
            role="Personalization Expert", 
            goal="Create personalized recommendations based on customer preferences",
            backstory="You specialize in matching customer preferences with destination options. You understand dietary restrictions, budget constraints, and personal interests.",
            verbose=False,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=60
        )
        
        # Agent 3: Itinerary Orchestrator Agent
        self.orchestrator_agent = Agent(
            llm=self.main_llm,
            role="Itinerary Planner",
            goal="Create practical, detailed day-by-day itineraries", 
            backstory="You create realistic travel itineraries that balance activities, meals, and travel time. You consider customer preferences and practical constraints.",
            verbose=False,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=60
        )
    
    def search_destination_info(self, destination: str, dates: List[str]) -> str:
        """Helper method to search for destination information"""
        search_queries = [
            f"{destination} weather June 2025",
            f"{destination} top attractions museums"
        ]
        
        all_results = []
        for query in search_queries:
            try:
                result = self.search_tool.search(query)
                all_results.append(f"Query: {query}\nResults:\n{result}\n")
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {str(e)}")
                all_results.append(f"Query: {query}\nResults: Search unavailable\n")
        
        return "\n=================\n".join(all_results)
    
    def plan_trip_chunked(self, customer_id: str, trip_context: TripContext) -> str:
        """Chunked trip planning to avoid truncation"""
        
        try:
            # Create context for RAG retrieval
            query_context = f"Planning trip to {trip_context.destination} for {trip_context.duration} days, {trip_context.trip_type} travel"
            
            # Retrieve customer profile using RAG
            customer_profile = self.rag_system.retrieve_customer_profile(customer_id, query_context)
            
            if not customer_profile:
                return "Customer profile not found. Please ensure customer is registered in the system."
            
            # Get destination research data
            logger.info("Gathering destination research...")
            destination_research = self.search_destination_info(trip_context.destination, trip_context.dates)
            
            # Generate each section separately with focused prompts
            sections = {}
            
            # Section 1: Destination Analysis
            dest_prompt = f"""Analyze {trip_context.destination} for June 2025. Based on this research:

{destination_research[:800]}

Provide a complete analysis covering:
1. Weather conditions and what to expect
2. Top 5 must-see attractions and museums
3. Local dining scene overview
4. Transportation tips
5. Cultural highlights

Be comprehensive and informative. Complete your response fully."""

            sections['destination'] = self.main_llm.invoke(dest_prompt)
            logger.info("Destination analysis completed")
            
            # Section 2: Personalized Recommendations
            personal_prompt = f"""Create detailed personalized recommendations for:

Customer: {customer_profile['name']}
Preferences: {customer_profile['preferences']}
Dietary Restrictions: {customer_profile['dietary_restrictions']}
Budget: {customer_profile['budget_range']}
Travel History: {customer_profile['travel_history']}

For {trip_context.destination}, recommend:
- 4 specific accommodation options with names and reasons
- 6 restaurants (considering dietary needs) with descriptions
- 8 activities/attractions matching interests with details

Be specific with names and explanations. Complete your full response."""

            sections['personalization'] = self.main_llm.invoke(personal_prompt)
            logger.info("Personalization completed")
            
            # Section 3: Daily Itinerary
            itinerary_prompt = f"""Create a detailed {trip_context.duration}-day itinerary:

Customer: {customer_profile['name']} ({customer_profile['budget_range']} budget)
Dates: {', '.join(trip_context.dates)}
Travelers: {trip_context.travelers}
Preferences: {customer_profile['preferences']}
Dietary: {customer_profile['dietary_restrictions']}

Create a complete day-by-day schedule including:
- Specific morning, afternoon, evening activities
- Restaurant recommendations for each meal
- Transportation between locations
- Timing and practical tips
- Alternative options for weather

Make it detailed and actionable. Complete the full {trip_context.duration}-day plan."""

            sections['itinerary'] = self.main_llm.invoke(itinerary_prompt)
            logger.info("Itinerary creation completed")
            
            # Section 4: Practical Tips
            tips_prompt = f"""Provide comprehensive travel tips for {customer_profile['name']}'s trip to {trip_context.destination}:

Budget: {customer_profile['budget_range']}
Dietary Restrictions: {customer_profile['dietary_restrictions']}
Duration: {trip_context.duration} days

Include:
- Packing recommendations for June weather
- Money-saving tips
- Cultural etiquette
- Safety considerations
- Transportation options
- Emergency contacts and useful phrases

Be thorough and complete your response."""

            sections['tips'] = self.main_llm.invoke(tips_prompt)
            logger.info("Practical tips completed")
            
            # Combine all sections
            final_result = f"""**COMPREHENSIVE TRAVEL PLAN FOR {customer_profile['name'].upper()}**

**DESTINATION ANALYSIS**
{sections['destination']}

**PERSONALIZED RECOMMENDATIONS**
{sections['personalization']}

**DETAILED {trip_context.duration}-DAY ITINERARY**
{sections['itinerary']}

**PRACTICAL TRAVEL TIPS**
{sections['tips']}

**TRIP SUMMARY**
Destination: {trip_context.destination}
Duration: {trip_context.duration} days ({', '.join(trip_context.dates)})
Travelers: {trip_context.travelers}
Budget: {customer_profile['budget_range']}
Special Requirements: {', '.join(customer_profile['dietary_restrictions']) if customer_profile['dietary_restrictions'] else 'None'}
"""
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in plan_trip_chunked: {str(e)}")
            return self.plan_trip_simple(customer_id, trip_context)
    
    def plan_trip_enhanced(self, customer_id: str, trip_context: TripContext) -> str:
        """Enhanced trip planning method using direct agent calls instead of CrewAI"""
        
        try:
            # Create context for RAG retrieval
            query_context = f"Planning trip to {trip_context.destination} for {trip_context.duration} days, {trip_context.trip_type} travel"
            
            # Retrieve customer profile using RAG
            customer_profile = self.rag_system.retrieve_customer_profile(customer_id, query_context)
            
            if not customer_profile:
                return "Customer profile not found. Please ensure customer is registered in the system."
            
            # Get destination research data
            logger.info("Gathering destination research...")
            destination_research = self.search_destination_info(trip_context.destination, trip_context.dates)
            
            # Step 1: Research Analysis (with longer token limit)
            research_prompt = f"""Analyze this destination information for {trip_context.destination} in June 2025:

{destination_research[:1200]}

Provide a comprehensive summary covering:
1. Weather conditions in June
2. Top 5 must-see attractions
3. Dining scene overview
4. Transportation tips

Provide a complete and detailed response."""

            research_result = self.main_llm.invoke(research_prompt)
            logger.info("Research analysis completed")
            
            # Step 2: Personalization 
            personalization_prompt = f"""Create detailed personalized recommendations for this customer:

Customer: {customer_profile['name']}
Preferences: {customer_profile['preferences']}
Dietary Restrictions: {customer_profile['dietary_restrictions']}
Budget: {customer_profile['budget_range']}
Past Travel: {customer_profile['travel_history'][:2]}

Destination: {trip_context.destination}

Recommend:
- 3 suitable accommodations with specific names and reasons
- 5 restaurants (considering dietary needs) with details
- 6 activities matching interests with explanations

Provide complete details and finish your response fully."""

            personalization_result = self.main_llm.invoke(personalization_prompt)
            logger.info("Personalization completed")
            
            # Step 3: Itinerary Creation
            itinerary_prompt = f"""Create a detailed {trip_context.duration}-day itinerary for {trip_context.destination}:

Customer: {customer_profile['name']} - {customer_profile['budget_range']} budget
Dates: {', '.join(trip_context.dates)}
Travelers: {trip_context.travelers}

Preferences: {customer_profile['preferences']}
Dietary: {customer_profile['dietary_restrictions']}

Research Insights:
{research_result[:300]}

Personalized Recommendations:
{personalization_result[:400]}

Create a complete day-by-day schedule with:
- Morning, afternoon, evening activities for each day
- Restaurant suggestions for meals
- Practical travel tips and timing
- Transportation recommendations

Provide a complete {trip_context.duration}-day itinerary. Finish your response completely."""

            itinerary_result = self.main_llm.invoke(itinerary_prompt)
            logger.info("Itinerary creation completed")
            
            # Combine results
            final_result = f"""**TRAVEL PLAN FOR {customer_profile['name'].upper()}**

**DESTINATION ANALYSIS**
{research_result}

**PERSONALIZED RECOMMENDATIONS**
{personalization_result}

**DETAILED ITINERARY**
{itinerary_result}

**TRIP SUMMARY**
Destination: {trip_context.destination}
Duration: {trip_context.duration} days
Budget: {customer_profile['budget_range']}
Special Requirements: {', '.join(customer_profile['dietary_restrictions']) if customer_profile['dietary_restrictions'] else 'None'}
"""
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in plan_trip_enhanced: {str(e)}")
            return self.plan_trip_simple(customer_id, trip_context)
    
    def plan_trip_simple(self, customer_id: str, trip_context: TripContext) -> str:
        """Simplified trip planning method with direct LLM calls"""
        
        try:
            # Create context for RAG retrieval
            query_context = f"Planning trip to {trip_context.destination} for {trip_context.duration} days, {trip_context.trip_type} travel"
            
            # Retrieve customer profile using RAG
            customer_profile = self.rag_system.retrieve_customer_profile(customer_id, query_context)
            
            if not customer_profile:
                return "Customer profile not found. Please ensure customer is registered in the system."
            
            # Get destination research data
            logger.info("Gathering destination research...")
            destination_research = self.search_destination_info(trip_context.destination, trip_context.dates)
            
            # Create a comprehensive prompt for direct LLM call
            comprehensive_prompt = f"""You are a travel planning expert. Create a detailed and complete travel itinerary based on the following information:

DESTINATION: {trip_context.destination}
DATES: {', '.join(trip_context.dates)}
DURATION: {trip_context.duration} days
TRAVELERS: {trip_context.travelers}
TRIP TYPE: {trip_context.trip_type}

CUSTOMER PROFILE:
Name: {customer_profile['name']}
Preferences: {customer_profile['preferences']}
Dietary Restrictions: {customer_profile['dietary_restrictions']}
Budget Range: {customer_profile['budget_range']}
Past Travel: {customer_profile['travel_history']}

DESTINATION RESEARCH:
{destination_research[:1000]}

Please provide a comprehensive response including:
1. A destination overview with weather and highlights
2. Recommended accommodations matching the customer's budget and preferences
3. Restaurant suggestions considering dietary restrictions
4. Complete daily itinerary with activities aligned with customer interests
5. Practical travel tips and recommendations

Make sure to complete your entire response without cutting off. Provide full details for each section."""

            # Direct LLM call
            logger.info("Making direct LLM call for trip planning...")
            result = self.main_llm.invoke(comprehensive_prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in plan_trip_simple: {str(e)}")
            return f"An error occurred during trip planning: {str(e)}"
    
    def plan_trip(self, customer_id: str, trip_context: TripContext) -> str:
        """Main method to orchestrate the travel planning process"""
        
        try:
            # Use the new chunked planning method for better results
            return self.plan_trip_chunked(customer_id, trip_context)
            
        except Exception as e:
            logger.error(f"Error in plan_trip: {str(e)}")
            # Fallback to enhanced planning
            logger.info("Falling back to enhanced planning method...")
            return self.plan_trip_enhanced(customer_id, trip_context)

def test_individual_components():
    """Test individual components before running full system"""
    try:
        print("Testing environment setup...")
        
        # Test API keys
        if not os.environ.get("WATSONX_APIKEY"):
            print("❌ WATSONX_APIKEY not found")
            return False
        else:
            print("✅ WATSONX_APIKEY found")
            
        if not os.environ.get("SERPER_API_KEY"):
            print("⚠️  SERPER_API_KEY not found - search will be limited")
        else:
            print("✅ SERPER_API_KEY found")
        
        print("\nTesting Watson LLM directly...")
        try:
            test_llm = WatsonxLLM(
                model_id="meta-llama/llama-3-3-70b-instruct",
                url=WATSONX_URL,
                params={
                    "decoding_method": "greedy", 
                    "max_new_tokens": 1000,
                    "temperature": 0,
                },
                project_id=PROJECT_ID,
            )
            test_response = test_llm.invoke("Say 'Watson LLM working' and then write a 3-sentence description of Barcelona.")
            print(f"✅ Watson LLM test: {test_response}")
        except Exception as e:
            print(f"❌ Watson LLM test failed: {str(e)}")
            return False
        
        print("\nTesting RAG system...")
        rag = RAGCustomerDatabase()
        profile = rag.retrieve_customer_profile("sarah_001", "Barcelona trip")
        if profile:
            print(f"✅ Retrieved profile for {profile['name']}")
        else:
            print("❌ Failed to retrieve profile")
            return False
            
        print("\nTesting search tool...")
        search = CustomSearchTool()
        result = search.search("Barcelona Spain weather")
        print(f"✅ Search returned {len(result)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False

def main():
    """Travel Concierge System"""
    
    # First test components
    if not test_individual_components():
        print("Component tests failed. Please check your setup.")
        return
    
    try:
        # Initialize the system
        print("\n" + "="*50)
        print("Initializing Travel Concierge System...")
        concierge = TravelConciergeSystem()
        
        # Create a trip context
        trip = TripContext(
            destination="Barcelona, Spain",
            dates=["2025-06-15", "2025-06-16", "2025-06-17", "2025-06-18"],
            duration=4,
            travelers=1,
            trip_type="leisure"
        )
        
        # Plan the trip for Sarah
        print("Planning trip for Sarah...")
        result = concierge.plan_trip("sarah_001", trip)
        
        print("\n" + "="*50)
        print("TRAVEL CONCIERGE RESULT")
        print("="*50)
        print(result)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Main execution error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
