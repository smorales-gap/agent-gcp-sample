# main.py
import os
import json
import sqlalchemy
from decimal import Decimal
from flask import Flask, request, jsonify
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration, Part, Content

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

        # We explicitly wrap the initial user prompt in a Content object.
        history = [Content(role="user", parts=[Part.from_text(user_prompt)])]

        # Set a limit to prevent infinite loops
        max_turns = 10
        for _ in range(max_turns):
            # Generate content with the current history and available tools
            response = model.generate_content(history, tools=[execute_sql_tool])
            candidate = response.candidates[0]

            # The model's complete response is a single turn. Append it.
            history.append(candidate.content)

            # Check if the model's response contains function calls.
            if not candidate.content.parts or not candidate.content.parts[0].function_call:
                # This is the final text response. Break the loop and return.
                return jsonify({"response": candidate.text})

            # --- Process Function Calls ---
            function_response_parts = []
            for part in candidate.content.parts:
                if part.function_call:
                    function_call = part.function_call
                    tool_name = function_call.name
                    
                    if tool_name == "execute_sql_query":
                        query = function_call.args.get("query", "")
                        tool_output = execute_sql_query(query)
                        
                        try:
                            parsed_output = json.loads(tool_output)
                            serializable_history = [h.to_dict() for h in history]
                            response_data = {"result": parsed_output,
                                             "chat_history": serializable_history}
                        except (json.JSONDecodeError, TypeError):
                            response_data = {"result": tool_output}

                        function_response_parts.append(Part.from_function_response(
                            name=tool_name,
                            response=response_data
                        ))
                    else:
                        function_response_parts.append(Part.from_function_response(
                            name=tool_name,
                            response={"error": f"Tool '{tool_name}' not found."}
                        ))
            
            # This is the correct way to provide tool results back to the model.
            history.append(Content(role="user", parts=function_response_parts))

        serializable_history = [h.to_dict() for h in history]
        return jsonify({
            "response": "The conversation has reached its maximum length.",
            "chat_history": serializable_history
        })

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": f"An internal error occurred. {e}"}), 500

if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))



























