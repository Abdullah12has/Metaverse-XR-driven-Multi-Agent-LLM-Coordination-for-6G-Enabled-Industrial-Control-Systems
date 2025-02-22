# main.py
import warnings
from datetime import datetime
from crew import Icsagents

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run(input_data):
    """
    Run the crew with provided input data.
    
    Args:
        input_data (dict): Dictionary containing 'topic' and 'context' keys
        
    Returns:
        dict: Result dictionary with success status and result/error
    """
    try:
        # Extract topic from input data with default
        topic = input_data.get('topic', 'MATHS')
        context_data = input_data.get('context', {})  # Default empty dict if no context

        # Initialize Icsagents with the context_data
        ics_crew = Icsagents(context_data)
        
        # Create and run the crew
        result = ics_crew.crew(topic).kickoff()

        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
