# main.py
import os
import json
import sqlalchemy
from flask import Flask, request, jsonify
from google.cloud import aiplatform
from google.cloud.aiplatform.generative_models import (
    GenerativeModel,
    Tool,
    FunctionDeclaration,
    Schema,
)

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
    sqlalchemy.engine.url.URL.create(
        drivername="postgresql+psycopg2", 
        username=db_user,
        password=db_pass,
        database=db_name,
        query={"unix_sock": f"/cloudsql/{db_host}/.s.PGSQL.5432"}
    )
)

# --- Vertex AI configuration ---
# Replace with your GCP project and location
project_id = os.environ.get("GCP_PROJECT_ID")
location = os.environ.get("GCP_LOCATION", "us-central1")
aiplatform.init(project=project_id, location=location)

# Initialize the Gemini model with tool-calling capabilities
model = GenerativeModel("gemini-1.5-flash-preview-05-20")

# --- Agent Tool Definitions ---
def execute_sql_query(query: str):
    """Executes a read-only SQL query against the database and returns the results as a JSON string."""
    try:
        with engine.connect() as conn:
            app.logger.info(f"Executing SQL query: {query}")
            result = conn.execute(sqlalchemy.text(query))
            rows = [dict(row._mapping) for row in result.fetchall()]
            return json.dumps(rows)
    except Exception as e:
        app.logger.error(f"SQL execution error: {e}")
        return f"Error executing query: {e}"

# --- Define the tool for the LLM ---
execute_sql_tool = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="execute_sql_query",
            description="Executes a read-only SQL query against a products database and returns the results. Use this tool when the user asks for specific product information, like names, categories, or prices. Do not perform any write operations (e.g., INSERT, UPDATE, DELETE). The table schema is products(product_id INT, product_name VARCHAR, category VARCHAR, price DECIMAL).",
            parameters=Schema(
                type=Schema.Type.OBJECT,
                properties={
                    "query": Schema(
                        type=Schema.Type.STRING,
                        description="The complete SQL query to execute. It must be a SELECT statement."
                    )
                },
                required=["query"]
            )
        )
    ]
)

# --- Flask Endpoint for the Agent ---
@app.route("/agent", methods=["POST"])
def agent():
    """Main endpoint for the LLM agent that handles tool-calling."""
    try:
        user_prompt = request.json.get("prompt")
        response = model.generate_content(user_prompt, tools=[execute_sql_tool])

        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            tool_name = function_call.name
            tool_args = function_call.args
            
            if tool_name == "execute_sql_query":
                tool_output_string = execute_sql_query(tool_args["query"])
                response_with_tool_output = model.generate_content([user_prompt, tool_output_string])
                return jsonify({"response": response_with_tool_output.text})
            else:
                return jsonify({"error": "LLM attempted to call an unknown tool."})
        else:
            return jsonify({"response": response.text})

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))