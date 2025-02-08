from gradio_client import Client

# Connect to your Space
client = Client("RamiIbrahim/tunisian-arabiz")

# Call your model with input
result = client.predict("mekla khayba yesser")  # Replace with your input
print(result)