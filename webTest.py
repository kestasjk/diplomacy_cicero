import http.client

# Establish a connection to google.com
connection = http.client.HTTPSConnection("www.webdiplomacy.net")

# Send a GET request for the root path
connection.request("GET", "/")

# Get the response from the server
response = connection.getresponse()

# Print the status and reason
print(f"Status: {response.status}, Reason: {response.reason}")

# Read and print the response body
data = response.read()
print(data.decode('utf-8'))

# Close the connection
connection.close()
