echo "Change webui/server.py's origins[0] to the IP address that npm starts on"
cd ./webui
uvicorn server:app --reload &
npm start
