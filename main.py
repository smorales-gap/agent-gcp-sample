# main.py
import os
import json
import sqlalchemy
from decimal import Decimal
from flask import Flask, request, jsonify
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration, Part

# --- Initialize Flask app ---
app = Flask(__name__)

# --- Database configuration ---
# Environment variables for Cloud SQL connection
db_user = os.environ.get("DB_USER")
db_pass = os.environ.get("DB_PASS")
db_name = os.environ.get("DB_NAME")
db_host = os.environ.get("INSTANCE_CONNECTION_NAME")

# Create a database engine using a unix socket connection
engine = sqlalchemy.create_engine(
    f"postgresql+psycopg2://{db_user}:{db_pass}@/{db_name}?host=/cloudsql/{db_host}"
)

# --- Vertex AI configuration ---
# Replace with your GCP project and location
project_id = os.environ.get("GCP_PROJECT_ID")
location = os.environ.get("GCP_LOCATION", "us-central1")
aiplatform.init(project=project_id, location=location)

# Initialize the Gemini model with tool-calling capabilities
model = GenerativeModel("gemini-2.5-flash")

# --- Agent Tool Definitions ---
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

def execute_sql_query(query: str):
    """Executes a read-only SQL query against the database and returns the results as a JSON string."""
    try:
        with engine.connect() as conn:
            app.logger.info(f"Executing SQL query: {query}")
            result = conn.execute(sqlalchemy.text(query))
            rows = [dict(row._mapping) for row in result.fetchall()]
            return json.dumps(rows, cls=DecimalEncoder)
    except Exception as e:
        app.logger.error(f"SQL execution error: {e}")
        return f"Error executing query: {e}"

# --- Define the tool for the LLM ---
execute_sql_tool = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="execute_sql_query",
            description="Executes a read-only SQL query against a products database and returns the results. Use this tool when the user asks for specific product information, like names, categories, or prices. Do not perform any write operations (e.g., INSERT, UPDATE, DELETE). The table schema is products(product_id INT, product_name VARCHAR, category VARCHAR, price DECIMAL).",
            parameters={
                "type":"object",
                "properties":{
                    "query": {
                        "type": "string",
                        "description": "The complete SQL query to execute. It must be a SELECT statement."
                    }
                },
                "required": ["query"]
            }
        )
    ]
)

# --- Flask Endpoint for the Agent ---
@app.route("/agent", methods=["POST"])
def agent():
    """Main endpoint for the LLM agent that handles multi-step tool-calling."""
    try:
        user_prompt = request.json.get("prompt")
        if not user_prompt:
            return jsonify({"error": "Prompt must not be empty."}), 400

        # Use a list to maintain the conversation history.
        history = [Part.from_text(user_prompt)]

        while True:
            # Generate content with the current history and available tools
            response = model.generate_content(history, tools=[execute_sql_tool])
            candidate = response.candidates[0]

            # The model's complete response, containing any function calls, is a single turn.
            # We append the entire turn (as a Content object) to the history.
            history.append(candidate.content)

            # Check if the model's response contains function calls.
            # If the first part isn't a function call, we assume it's the final text response.
            if not candidate.content.parts or not candidate.content.parts[0].function_call:
                # This is the final text response. Break the loop and return.
                return jsonify({"response": candidate.text})

            # --- Process Function Calls ---
            # Create a list to hold the responses for each function call in this turn.
            function_response_parts = []

            # Iterate through each part in the model's response.
            for part in candidate.content.parts:
                if part.function_call:
                    function_call = part.function_call
                    tool_name = function_call.name
                    
                    if tool_name == "execute_sql_query":
                        # Execute the tool and prepare the response data
                        query = function_call.args.get("query", "")
                        tool_output = execute_sql_query(query)
                        
                        try:
                            # The tool returns a JSON string; parse it for the response.
                            parsed_output = json.loads(tool_output)
                            response_data = {"result": parsed_output}
                        except (json.JSONDecodeError, TypeError):
                            # If output isn't valid JSON (e.g., an error message), use it directly.
                            response_data = {"result": tool_output}

                        # Append the processed tool output as a function response part.
                        function_response_parts.append(Part.from_function_response(
                            name=tool_name,
                            response=response_data
                        ))
                    else:
                        # Handle cases where the model calls an unknown tool.
                        function_response_parts.append(Part.from_function_response(
                            name=tool_name,
                            response={"error": f"Tool '{tool_name}' not found."}
                        ))
            
            # Add all the collected function responses to the history.
            # The SDK will interpret this sequence of response parts as the next single turn.
            history.extend(function_response_parts)
            
            # The loop will now continue, sending the tool results back to the model.

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": f"An internal error occurred. {e}"}), 500

if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))























