from src import create_app

print("Server is Running...")
print("Server on port : 8080 ")
app = create_app()

if __name__ == "__main__":
    print("Hello, World!")
    app.run(host="0.0.0.0", port=8080, debug=False)

