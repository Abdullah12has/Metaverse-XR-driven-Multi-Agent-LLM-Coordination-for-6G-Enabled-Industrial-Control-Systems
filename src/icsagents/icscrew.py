from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import pandas as pd
import json
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ICUOptimizationCrew(CrewBase):
    """CrewAI implementation for Industrial Control Unit optimization"""
    
    def __init__(self, context_data: Dict = None):
        """
        Initialize the ICU optimization crew.
        
        Args:
            context_data (Dict): Additional context data including sensor readings and current settings
        """
        self.context_data = context_data or {}
        self.ollama_llm = LLM(
            model='ollama/tinyllama',
            base_url='http://ollama:11434/api/generate',
        )
        
        # Define role prompts
        self.SENSOR_ANALYST_PROMPT = """
        You are an expert industrial sensor data analyst. Your responsibilities:
        1. Analyze real-time sensor data from industrial control units
        2. Identify patterns and anomalies in sensor readings
        3. Evaluate current performance metrics
        4. Make preliminary recommendations for optimization
        
        Current context: {context}
        """
        
        self.OPTIMIZATION_ENGINEER_PROMPT = """
        You are an experienced industrial optimization engineer. Your responsibilities:
        1. Review sensor analysis and current settings
        2. Calculate optimal control parameters
        3. Simulate proposed changes impact
        4. Generate detailed adjustment recommendations
        
        Current context: {context}
        """
        
        self.VALIDATION_EXPERT_PROMPT = """
        You are a validation expert for industrial systems. Your responsibilities:
        1. Verify proposed setting changes against safety parameters
        2. Assess potential risks and side effects
        3. Validate optimization recommendations
        4. Provide final approved settings with safety margins
        
        Current context: {context}
        """

    @agent
    def sensor_analyst(self) -> Agent:
        """Creates the sensor data analysis agent"""
        return Agent(
            name="Sensor Analyst",
            role="Industrial Sensor Data Analyst",
            goal="Analyze sensor data and identify optimization opportunities",
            backstory=self.SENSOR_ANALYST_PROMPT.format(context=self.context_data),
            llm=self.ollama_llm,
            verbose=True
        )

    @agent
    def optimization_engineer(self) -> Agent:
        """Creates the optimization engineer agent"""
        return Agent(
            name="Optimization Engineer",
            role="Industrial Optimization Expert",
            goal="Determine optimal control settings based on analysis",
            backstory=self.OPTIMIZATION_ENGINEER_PROMPT.format(context=self.context_data),
            llm=self.ollama_llm,
            verbose=True
        )

    @agent
    def validation_expert(self) -> Agent:
        """Creates the validation expert agent"""
        return Agent(
            name="Validation Expert",
            role="Safety and Validation Specialist",
            goal="Ensure proposed changes meet all safety requirements",
            backstory=self.VALIDATION_EXPERT_PROMPT.format(context=self.context_data),
            llm=self.ollama_llm,
            verbose=True
        )

    @task
    def analyze_sensor_data(self) -> Task:
        """Task for analyzing current sensor readings"""
        return Task(
            description="""
            1. Review all current sensor readings
            2. Compare with historical patterns
            3. Identify performance gaps
            4. Generate analysis report
            """,
            agent=self.sensor_analyst,
            context=self.context_data
        )

    @task
    def optimize_settings(self) -> Task:
        """Task for determining optimal settings"""
        return Task(
            description="""
            1. Review sensor analysis
            2. Calculate optimal parameters
            3. Model expected improvements
            4. Generate detailed recommendations
            """,
            agent=self.optimization_engineer,
            context=self.context_data
        )

    @task
    def validate_changes(self) -> Task:
        """Task for validating proposed changes"""
        return Task(
            description="""
            1. Review proposed setting changes
            2. Verify safety compliance
            3. Assess operational risks
            4. Provide final approved settings
            """,
            agent=self.validation_expert,
            context=self.context_data
        )

    @crew
    def crew(self) -> Crew:
        """Assembles the ICU optimization crew"""
        return Crew(
            agents=[
                self.sensor_analyst,
                self.optimization_engineer,
                self.validation_expert
            ],
            tasks=[
                self.analyze_sensor_data(),
                self.optimize_settings(),
                self.validate_changes()
            ],
            process=Process.sequential,
            verbose=True
        )

    def train(self, training_file: str) -> None:
        """
        Train the crew using historical data from a text file.
        
        Args:
            training_file (str): Path to training data file
        """
        try:
            # Load historical data
            with open(training_file, 'r') as f:
                training_data = f.readlines()
            
            # Parse and structure training data
            parsed_data = []
            for line in training_data:
                try:
                    data_point = json.loads(line.strip())
                    parsed_data.append(data_point)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid data line: {line}")
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(parsed_data)
            
            logger.info(f"Training crew on {len(df)} historical data points")
            
            # Use CrewAI's train function for each agent
            self.sensor_analyst().train(df)
            self.optimization_engineer().train(df)
            self.validation_expert().train(df)
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise

def run_optimization(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the ICU optimization process with provided input data.
    
    Args:
        input_data (Dict[str, Any]): Dictionary containing current ICU readings and settings
        
    Returns:
        Dict[str, Any]: Optimization results including new settings and expected improvements
    """
    try:
        # Validate input data
        required_fields = ['sensor_readings', 'current_settings', 'safety_limits']
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Initialize crew with input data
        icu_crew = ICUOptimizationCrew(context_data=input_data)
        
        # Run optimization process
        result = icu_crew.crew().kickoff()
        
        # Parse and validate results
        try:
            optimization_result = json.loads(result)
            return {
                "success": True,
                "new_settings": optimization_result.get("settings"),
                "expected_improvements": optimization_result.get("improvements"),
                "safety_verification": optimization_result.get("safety_check"),
                "timestamp": datetime.now().isoformat()
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Invalid optimization result format",
                "raw_result": result
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    # Example input data
    sample_input = {
        "sensor_readings": {
            "temperature": 85.5,
            "pressure": 2.3,
            "flow_rate": 150.2
        },
        "current_settings": {
            "valve_position": 65,
            "pump_speed": 80,
            "temperature_setpoint": 85
        },
        "safety_limits": {
            "max_temperature": 95,
            "max_pressure": 3.0,
            "max_flow_rate": 200
        }
    }
    
    # Run optimization
    result = run_optimization(sample_input)
    print(json.dumps(result, indent=2))
    
    # Train crew (example)
    # crew = ICUOptimizationCrew()
    # crew.train("historical_data.txt")