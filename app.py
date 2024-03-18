from src import create_app
from waitress import serve

print("Server is Running...")
print("Server on port : 8080 ")
app = create_app()
serve(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    print("Hello, World!")
    app.run(debug=False)
