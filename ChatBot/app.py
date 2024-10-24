from flask import Flask, request, jsonify
import os
import pandas as pd
from transformers import pipeline
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.utils.generic')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Create the Flask app
app = Flask(__name__)

# Initialize the Bot Framework Adapter with your Bot ID and Password from Azure
adapter_settings = BotFrameworkAdapterSettings("0c68caca-4e89-477e-ba83-1f66b299f2b1", "juG8Q~cuL6sCq3jyZXmP1-4.xVXttzTcXIMYvbJY")
adapter = BotFrameworkAdapter(adapter_settings)

# Load the NLP model at the start of the application
def load_nlp_model():
    return pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

nlp_model = load_nlp_model()

# Load the dataset
def load_data():
    try:
        data = pd.read_csv('custdata.csv')
        if data.empty:
            return {"error": "The data file is empty."}, 400
        
        data.columns = data.columns.str.strip()  # Clean column names
        return data
    except FileNotFoundError:
        return {"error": "The data file 'custdata.csv' was not found."}, 404
    except pd.errors.EmptyDataError:
        return {"error": "The data file is empty. Please check the contents."}, 400
    except pd.errors.ParserError:
        return {"error": "Error parsing the data file. Please check its format."}, 400
    except Exception as e:
        return {"error": f"An error occurred while loading the data: {e}"}, 500

# Generate responses for multiple questions
def get_responses(user_input, cust_id, data):
    customer_data = data[data['cust_id'].astype(str) == str(cust_id)]
    
    if customer_data.empty:
        return ["The customer ID you provided does not exist."]

    product_name = customer_data['product_name'].values[0]
    shipment_status = customer_data['shipment_progress'].values[0]
    purchase_date = customer_data['purchase_date'].values[0]
    price = customer_data['price'].values[0]

    # Prepare the context for the model
    context = f"""
    You ordered: {product_name}.
    Shipment status: {shipment_status}.
    Purchase date: {purchase_date}.
    Price: ${price:.2f}.
    """

    responses = []
    questions = user_input.split('?')  # Split by question mark

    for question in questions:
        question = question.strip()  # Clean up whitespace
        if not question:
            continue
        
        # Check for specific questions
        if "bill" in question.lower() or "price" in question.lower():
            responses.append(f"The price of your order is: ${price:.2f}.")
        elif "when did I order" in question.lower() or "purchase date" in question.lower():
            responses.append(f"You purchased this on: {purchase_date}.")
        else:
            # Get the model's response for other questions
            response = nlp_model(question=question, context=context)
            if response['score'] > 0.1:
                responses.append(response['answer'])
            else:
                responses.append("I'm sorry, I couldn't understand your question.")

    return responses

# Route for Bot to handle incoming messages
@app.route("/api/messages", methods=["POST"])
def messages():
    if "application/json" in request.headers["Content-Type"]:
        json_message = request.json
    else:
        return jsonify({"status": "Invalid request"}), 400

    activity = Activity().deserialize(json_message)

    async def turn_call(turn_context: TurnContext):
        data = load_data()
        user_input = activity.text
        cust_id = activity.from_property.id  # Assuming user ID is the customer ID
        responses = get_responses(user_input, cust_id, data)
        await turn_context.send_activity(f"{' '.join(responses)}")

    task = adapter.process_activity(activity, "", turn_call)
    return jsonify({"status": "Message received"}), 200

# Run the Flask application
if __name__ == "__main__":
    app.run(port=3978)
