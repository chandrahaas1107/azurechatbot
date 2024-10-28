from flask import Flask, request, jsonify
import os
import pandas as pd
from transformers import pipeline
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity
import warnings
import tensorflow as tf

# Suppress specific warnings from TensorFlow and transformers
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.utils.generic')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Initialize Flask application
app = Flask(__name__)

# Bot Framework Adapter settings - use your Azure Bot ID and Password
BOT_ID = "0c68caca-4e89-477e-ba83-1f66b299f2b1"
BOT_PASSWORD = "juG8Q~cuL6sCq3jyZXmP1-4.xVXttzTcXIMYvbJY"
adapter_settings = BotFrameworkAdapterSettings(BOT_ID, BOT_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

# Load NLP model at the start of the application
def load_nlp_model():
    return pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

nlp_model = load_nlp_model()

# Function to load customer data from a CSV file
def load_data():
    try:
        data = pd.read_csv('custdata.csv')
        if data.empty:
            return {"error": "The data file is empty."}, 400
        
        # Clean column names
        data.columns = data.columns.str.strip()
        return data
    except FileNotFoundError:
        return {"error": "The data file 'custdata.csv' was not found."}, 404
    except pd.errors.EmptyDataError:
        return {"error": "The data file is empty. Please check the contents."}, 400
    except pd.errors.ParserError:
        return {"error": "Error parsing the data file. Please check its format."}, 400
    except Exception as e:
        return {"error": f"An error occurred while loading the data: {e}"}, 500

# Function to generate responses based on user input
def get_responses(user_input, cust_id, data):
    # Filter customer data by customer ID
    customer_data = data[data['cust_id'].astype(str) == str(cust_id)]
    
    if customer_data.empty:
        return ["The customer ID you provided does not exist."]

    product_name = customer_data['product_name'].values[0]
    shipment_status = customer_data['shipment_progress'].values[0]
    purchase_date = customer_data['purchase_date'].values[0]
    price = customer_data['price'].values[0]

    # Prepare the context for the NLP model
    context = f"""
    You ordered: {product_name}.
    Shipment status: {shipment_status}.
    Purchase date: {purchase_date}.
    Price: ${price:.2f}.
    """

    responses = []
    questions = [q.strip() for q in user_input.split('?') if q.strip()]

    for question in questions:
        # Specific questions handling
        if "bill" in question.lower() or "price" in question.lower():
            responses.append(f"The price of your order is: ${price:.2f}.")
        elif "when did I order" in question.lower() or "purchase date" in question.lower():
            responses.append(f"You purchased this on: {purchase_date}.")
        else:
            # Get response from NLP model for other questions
            response = nlp_model(question=question, context=context)
            if response['score'] > 0.1:
                responses.append(response['answer'])
            else:
                responses.append("I'm sorry, I couldn't understand your question.")

    return responses

# Route for handling bot messages
@app.route("/api/messages", methods=["POST"])
def messages():
    if request.headers.get("Content-Type") == "application/json":
        json_message = request.get_json()
    else:
        return jsonify({"status": "Invalid request"}), 400

    activity = Activity().deserialize(json_message)

    async def turn_call(turn_context: TurnContext):
        data = load_data()
        if isinstance(data, tuple):
            # If there's an error loading the data, return the error
            await turn_context.send_activity(f"{data[0]['error']}")
            return

        user_input = activity.text
        cust_id = activity.from_property.id  # Assuming user ID is the customer ID
        responses = get_responses(user_input, cust_id, data)
        await turn_context.send_activity(f"{' '.join(responses)}")

    task = adapter.process_activity(activity, "", turn_call)
    return jsonify({"status": "Message received"}), 200

# Run the Flask application
if __name__ == "__main__":
    # Run Flask app on host 0.0.0.0 to be accessible in Azure
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
