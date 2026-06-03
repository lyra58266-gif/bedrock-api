from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

def permutation_entropy(ts, dim=3, tau=1):
    n = len(ts)
    if n < dim * tau: return np.nan
    embedded = np.array([ts[i:i+dim*tau:tau] for i in range(n-dim*tau+1)])
    patterns = np.apply_along_axis(lambda x: np.argsort(x).tobytes(), 1, embedded)
    _, counts = np.unique(patterns, return_counts=True)
    probs = counts / len(patterns)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def largest_lyap_rosenstein(ts, emb_dim=3, tau=1, min_tsep=10):
    n = len(ts)
    if n < emb_dim*tau + min_tsep + 50: return np.nan
    N = n - (emb_dim-1)*tau
    X = np.array([ts[i:i+emb_dim*tau:tau] for i in range(N)])
    tree = KDTree(X)
    lyap_sum, count = 0.0, 0
    for i in range(int(0.1*N), N-min_tsep):
        dist, idx = tree.query(X[i], k=2, p=2)
        j = idx[1] if idx[0]==i else idx[0]
        d0 = dist[1] if idx[0]==i else dist[0]
        if abs(i-j) < min_tsep: continue
        if i+1 < N and j+1 < N:
            d1 = np.linalg.norm(X[i+1]-X[j+1])
            if d0>0 and d1>0:
                lyap_sum += np.log(d1/d0)
                count += 1
        if count > 100: break
    return lyap_sum/count if count>0 else np.nan

def compute_bedrock(ticker, years=5):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty or len(data) < 150:
        return None, None, None, "Not enough data."
    prices = data['Close'].values.flatten()
    log_returns = np.diff(np.log(prices))
    log_returns = log_returns[~np.isnan(log_returns)]
    if len(log_returns) < 100: return None, None, None, "Not enough data."
    window, step = min(150, len(log_returns)//2), 15
    lyap_vals, pe_vals = [], []
    for i in range(0, len(log_returns)-window, step):
        local = log_returns[i:i+window]
        if len(local) < 100: continue
        lyap_vals.append(largest_lyap_rosenstein(local, emb_dim=3, tau=1, min_tsep=20))
        pe_vals.append(permutation_entropy(local, dim=3, tau=1))
    lyap_vals = np.array(lyap_vals, dtype=float)
    pe_vals = np.array(pe_vals, dtype=float)
    mask = ~np.isnan(lyap_vals) & ~np.isnan(pe_vals)
    lyap_vals, pe_vals = lyap_vals[mask], pe_vals[mask]
    if len(lyap_vals) < 2: return None, None, None, "Computation failed."
    sigma = 0.02
    lyap_danger = np.exp(-(lyap_vals**2)/(2*sigma**2))
    pe_norm = (pe_vals - np.nanmin(pe_vals)) / (np.nanmax(pe_vals) - np.nanmin(pe_vals))
    fragility = 0.5*lyap_danger + 0.5*pe_norm
    bedrock = 1.0 - fragility[-1]
    recent_returns = np.mean(log_returns[-20:]) if len(log_returns) >= 20 else 0.0
    return bedrock, lyap_vals[-1], recent_returns, None

def generate_commentary(ticker, bedrock, recent_returns):
    if recent_returns > 0.02: price_moves, price_dir = "rising strongly", "up"
    elif recent_returns > 0: price_moves, price_dir = "slightly up", "up"
    elif recent_returns > -0.02: price_moves, price_dir = "slightly down", "down"
    else: price_moves, price_dir = "declining notably", "down"
    if bedrock > 0.60 and price_dir == "up":
        status, commentary = "HEALTHY", f"{ticker} shows high internal cohesion (Bedrock = {bedrock:.2f}). The price is {price_moves}, and this aligns with the strong foundation. No contradiction. This is the safest scenario."
    elif bedrock > 0.60 and price_dir == "down":
        status, commentary = "TRANSIENT", f"Despite the price {price_moves}, {ticker} maintains strong internal cohesion (Bedrock = {bedrock:.2f}). This is a positive contradiction. The decline appears to be a correction or passing fear, not a structural collapse."
    elif bedrock < 0.50 and price_dir == "up":
        status, commentary = "BUBBLE_WARNING", f"Caution: {ticker} shows internal fragility (Bedrock = {bedrock:.2f}), yet the price is {price_moves}. This is a warning contradiction. The rally lacks structural support."
    elif bedrock < 0.50 and price_dir == "down":
        status, commentary = "CRITICAL", f"Alert: {ticker} is internally fragile (Bedrock = {bedrock:.2f}) and the price is {price_moves}. This is the worst scenario. The drop appears structural."
    else:
        status, commentary = "NEUTRAL", f"{ticker} is in a moderate cohesion state (Bedrock = {bedrock:.2f}). The price is {price_moves}. No strong contradiction is present."
    return status, commentary

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bedrock — Cohesion Index</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 600px; margin: 5% auto; padding: 0 20px; background: #fafafa; color: #1a1a1a; }
        h1 { font-size: 2.5rem; margin-bottom: 0.2rem; }
        .subtitle { color: #555; margin-bottom: 2rem; }
        input { width: 70%; padding: 12px; font-size: 1.2rem; border: 1px solid #ccc; border-radius: 6px; }
        button { padding: 12px 24px; font-size: 1.2rem; background: #1a1a1a; color: white; border: none; border-radius: 6px; cursor: pointer; }
        .result { margin-top: 2rem; padding: 1.5rem; border-radius: 8px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        .bedrock-value { font-size: 3rem; font-weight: bold; }
        .status { font-size: 1.2rem; margin: 0.5rem 0; }
        .commentary { margin-top: 1rem; line-height: 1.6; }
        .disclaimer { color: #888; font-size: 0.85rem; margin-top: 1.5rem; border-top: 1px solid #eee; padding-top: 1rem; }
    </style>
</head>
<body>
    <h1>Bedrock</h1>
    <p class="subtitle">Cohesion Index — Not a prediction. A whisper.</p>
    <input type="text" id="ticker" placeholder="Enter ticker (e.g., MSFT)" value="MSFT">
    <button onclick="checkBedrock()">Check</button>
    <div id="result" class="result" style="display:none;">
        <div class="bedrock-value" id="bedrockValue"></div>
        <div class="status" id="bedrockStatus"></div>
        <div class="commentary" id="bedrockCommentary"></div>
        <div class="disclaimer" id="bedrockDisclaimer"></div>
    </div>
    <script>
        async function checkBedrock() {
            const ticker = document.getElementById('ticker').value.toUpperCase();
            const res = await fetch('/api/v1/bedrock?ticker=' + ticker);
            const data = await res.json();
            document.getElementById('result').style.display = 'block';
            if (data.error) {
                document.getElementById('bedrockValue').innerText = 'Error';
                document.getElementById('bedrockCommentary').innerText = data.error;
                return;
            }
            document.getElementById('bedrockValue').innerText = data.bedrock;
            document.getElementById('bedrockStatus').innerText = 'Status: ' + data.status;
            document.getElementById('bedrockCommentary').innerText = data.commentary;
            document.getElementById('bedrockDisclaimer').innerText = data.disclaimer;
        }
        window.onload = checkBedrock;
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/api/v1/bedrock', methods=['GET'])
def bedrock_endpoint():
    ticker = request.args.get('ticker', 'MSFT').upper().strip()
    try:
        bedrock, current_lyap, recent_ret, error = compute_bedrock(ticker)
        if error: return jsonify({"ticker": ticker, "error": error}), 400
        status, commentary = generate_commentary(ticker, bedrock, recent_ret)
        return jsonify({
            "ticker": ticker,
            "bedrock": round(bedrock, 3),
            "status": status,
            "commentary": commentary,
            "disclaimer": "Bedrock measures internal structural cohesion. It is not financial advice. It does not predict price direction. Decisions are yours alone."
        })
    except Exception as e:
        return jsonify({"ticker": ticker, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "Bedrock API v1.0"})
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
