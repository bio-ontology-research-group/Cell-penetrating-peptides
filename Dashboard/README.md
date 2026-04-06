# CPP Mechanisms KG Dashboard — Flask

## Quick Start on cbontsr01.kaust.edu.sa

### 1. Copy files to the server
```bash
scp -r kg_dashboard_flask/ user@cbontsr01.kaust.edu.sa:~/
```

### 2. SSH into the server
```bash
ssh user@cbontsr01.kaust.edu.sa
cd ~/kg_dashboard_flask
```

### 3. Set up Python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. (Optional) Copy your local TTL file
If you have a local `mechanisms.ttl`, place it in the project directory:
```bash
cp /path/to/mechanisms.ttl ~/kg_dashboard_flask/mechanisms.ttl
# or inside an Ontology subfolder:
mkdir -p ~/kg_dashboard_flask/Ontology
cp /path/to/mechanisms.ttl ~/kg_dashboard_flask/Ontology/mechanisms.ttl
```
If no local file is found, the app will download it from GitHub automatically.

### 5. Run the app
```bash
python app.py
```
The dashboard will be available at: **http://cbontsr01.kaust.edu.sa:5001**

---

## Running as a background service (recommended for production)

### Option A: nohup (simple)
```bash
nohup python app.py > app.log 2>&1 &
echo "PID: $!"
```

### Option B: systemd service
Create `/etc/systemd/system/kg-dashboard.service`:
```ini
[Unit]
Description=CPP KG Dashboard
After=network.target

[Service]
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/kg_dashboard_flask
ExecStart=/home/YOUR_USERNAME/kg_dashboard_flask/venv/bin/python app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable kg-dashboard
sudo systemctl start kg-dashboard
sudo systemctl status kg-dashboard
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main dashboard UI |
| GET | `/api/status` | Graph load status |
| GET | `/api/metrics` | Global triple/node/predicate counts |
| GET | `/api/classes` | Instance counts per class |
| GET | `/api/associations` | Association predicate usage counts |
| POST | `/api/search` | Peptide search (`{"field":"sequence","term":"RQIK"}`) |
| POST | `/api/sparql` | Run SPARQL query (`{"query":"SELECT..."}`) |

## File Structure
```
kg_dashboard_flask/
├── app.py              # Flask backend
├── requirements.txt
├── README.md
├── templates/
│   └── index.html      # Full frontend (Bootstrap + Chart.js)
└── Ontology/           # (optional) place mechanisms.ttl here
    └── mechanisms.ttl
```
