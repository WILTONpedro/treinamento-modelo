import requests

url = "http://localhost:5000/classificar"
data = {"texto": "Experiência com vendas e atendimento ao cliente."}

response = requests.post(url, json=data)  # note o `json=data` aqui
print(response.json())