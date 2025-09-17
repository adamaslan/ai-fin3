# GCP Free Tier Stock Prediction Pipeline - Complete Implementation Guide

## Current Setup Configuration

### Infrastructure
- **VM Instance**: ttb1-machine-sept (e2-micro, Debian)
- **Storage Bucket**: ttb-bucket1
- **Virtual Environment**: ~/stock_env
- **User**: chillcoders
- **Models**: Scikit-learn Ensemble (RandomForest + GradientBoosting + LogisticRegression)

### Optimized Resource Usage (PyTorch-Free Configuration)

| Resource | Free Tier Limit | Current Usage | Percentage Used | Status |
|----------|----------------|---------------|-----------------|---------|
| **Compute Engine** | 744 hours/month | ~2 hours/month | <1% | Safe |
| **Cloud Storage** | 5 GB | ~7 MB/month | <1% | Safe |
| **Data Transfer** | 1 GB/month | ~110 MB/month | 11% | Safe |
| **API Operations** | 5,000 Class A | ~22/month | <1% | Safe |
| **Persistent Disk** | 30 GB | ~1 GB used | 3% | Safe |

## Implementation Steps Completed

### 1. Environment Setup (PyTorch-Free)
```bash
# Virtual environment creation
python3 -m venv ~/stock_env
source ~/stock_env/bin/activate

# Lightweight package installation (no PyTorch)
pip install yfinance pandas numpy scikit-learn google-cloud-storage
```

### 2. Enhanced Script Features
- **File**: `stock_predictor.py` 
- **Stocks Tracked**: AAPL, MSFT, GOOGL, TSLA, NVDA
- **Cache Directory**: ~/stock_cache (7-day cache)
- **Backup Directory**: ~/stock_predictions
- **Data Period**: 3 months per stock
- **Memory Usage**: ~150-200MB (down from 1GB+ with PyTorch)

### 3. Automation Schedule
```bash
# Cron job for Monday-Friday at 10 AM EST (15:00 UTC)
(crontab -l 2>/dev/null; echo "0 15 * * 1-5 cd /home/chillcoders && source stock_env/bin/activate && python stock_predictor.py >> predictions.log 2>&1") | crontab -
```

## Enhanced Free Tier GCP Services Integration

### Additional Free Tier Services Available

#### Cloud Functions (2M Invocations/Month)
```python
# Optional webhook for real-time alerts
def send_prediction_alert(request):
    """Cloud Function to send email alerts for strong buy/sell signals"""
    # Trigger: Pub/Sub message from stock predictor
    # Action: Send email via SendGrid (free tier: 100 emails/day)
```

#### Cloud Scheduler (3 Jobs Free)
```yaml
# Alternative to cron - managed scheduling
name: "stock-prediction-job"
schedule: "0 15 * * 1-5"  # Mon-Fri 10 AM EST
time_zone: "America/New_York"
target:
  uri: "https://your-cloud-function-url"
```

#### Firestore (1 GiB Storage, 50K Reads/Day)
```python
# Store predictions in NoSQL database for easy querying
def save_to_firestore(predictions):
    from google.cloud import firestore
    db = firestore.Client()
    
    for symbol, data in predictions.items():
        doc_ref = db.collection('predictions').document(f"{symbol}_{date}")
        doc_ref.set({
            'symbol': symbol,
            'price': data['current_price'],
            'signal': data['signal'],
            'confidence': data['confidence'],
            'timestamp': datetime.now()
        })
```

#### Cloud Monitoring (Free Metrics & Alerts)
```python
# Monitor VM performance and prediction accuracy
from google.cloud import monitoring_v3

def create_custom_metric(prediction_accuracy):
    client = monitoring_v3.MetricServiceClient()
    # Track prediction success rates over time
    # Alert if accuracy drops below threshold
```

#### BigQuery (1 TB Queries/Month, 10 GB Storage)
```sql
-- Analyze historical prediction performance
CREATE TABLE stock_analysis.predictions AS
SELECT 
    symbol,
    prediction_date,
    predicted_signal,
    actual_outcome_3d,
    accuracy_score
FROM stock_predictions
WHERE prediction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
```

#### Secret Manager (6 Active Secrets Free)
```python
# Secure API key storage
from google.cloud import secretmanager

def get_api_keys():
    client = secretmanager.SecretManagerServiceClient()
    alpha_vantage_key = client.access_secret_version(
        request={"name": "projects/PROJECT_ID/secrets/alpha-vantage-key/versions/latest"}
    ).payload.data.decode("UTF-8")
    return alpha_vantage_key
```

### Enhanced Architecture with Free Tier Services

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cloud         │    │   Compute       │    │   Cloud         │
│   Scheduler     │───▶│   Engine        │───▶│   Storage       │
│   (3 jobs free) │    │   (e2-micro)    │    │   (5 GB)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Firestore     │    │   Cloud         │
                       │   (1 GiB free)  │    │   Functions     │
                       └─────────────────┘    │   (2M/month)    │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   Cloud         │
                                              │   Monitoring    │
                                              │   (Free alerts) │
                                              └─────────────────┘
```

### Resource Usage with Enhanced Services

| Service | Free Tier Limit | Projected Usage | Percentage | Enhancement |
|---------|----------------|-----------------|------------|-------------|
| **Compute Engine** | 744 hours/month | ~2 hours/month | <1% | Core processing |
| **Cloud Storage** | 5 GB | ~10 MB/month | <1% | Raw data & backups |
| **Firestore** | 1 GiB, 50K reads/day | ~1 MB, 100 reads/day | <1% | Structured predictions |
| **Cloud Functions** | 2M invocations/month | ~50/month | <1% | Alert processing |
| **BigQuery** | 1 TB queries, 10 GB storage | 100 MB queries, 50 MB storage | <1% | Analytics |
| **Secret Manager** | 6 active secrets | 3 secrets | 50% | API key storage |

## Scaling to 50 Stocks with Enhanced Free Tier

### Multi-Service Architecture
```python
class EnhancedFreeTierPredictor:
    def __init__(self):
        self.batch_schedule = {
            'monday': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE'],
            'tuesday': ['JPM', 'JNJ', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'INTC', 'VZ'],
            'wednesday': ['BAC', 'KO', 'PFE', 'XOM', 'WMT', 'MRK', 'T', 'CVX', 'ABT', 'TMO'],
            'thursday': ['COST', 'AVGO', 'ACN', 'DHR', 'TXN', 'NEE', 'LIN', 'PM', 'HON', 'UPS'],
            'friday': ['LOW', 'QCOM', 'C', 'BMY', 'MDT', 'RTX', 'SCHW', 'GS', 'CAT', 'DE']
        }
        self.firestore_client = firestore.Client()
        self.storage_client = storage.Client()
```

### 50-Stock Resource Impact

| Resource | Free Tier Limit | 50-Stock Usage | Percentage | Buffer |
|----------|----------------|----------------|------------|--------|
| **Data Transfer** | 1 GB/month | ~550 MB/month | 55% | 450 MB |
| **Cloud Storage** | 5 GB | ~35 MB/month | <1% | 4.97 GB |
| **Firestore Reads** | 50K/day | ~250/day | <1% | 49.75K |
| **BigQuery Storage** | 10 GB | ~100 MB | 1% | 9.9 GB |

### Advanced Features with Free Tier Services

#### 1. Real-Time Alerting System
```python
def trigger_alert_function(strong_signals):
    """Trigger Cloud Function for email alerts"""
    from google.cloud import pubsub_v1
    
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, 'stock-alerts')
    
    message_data = json.dumps({
        'signals': strong_signals,
        'timestamp': datetime.now().isoformat()
    }).encode('utf-8')
    
    publisher.publish(topic_path, message_data)
```

#### 2. Historical Performance Analytics
```python
def analyze_prediction_accuracy():
    """Use BigQuery to analyze historical accuracy"""
    from google.cloud import bigquery
    
    client = bigquery.Client()
    query = """
    SELECT 
        symbol,
        AVG(CASE WHEN predicted_signal = actual_outcome THEN 1 ELSE 0 END) as accuracy,
        COUNT(*) as prediction_count
    FROM stock_analysis.predictions
    WHERE prediction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
    GROUP BY symbol
    ORDER BY accuracy DESC
    """
    
    results = client.query(query)
    return list(results)
```

#### 3. Adaptive Model Selection
```python
def adaptive_model_selection(symbol, historical_performance):
    """Choose best model based on historical accuracy per symbol"""
    if historical_performance[symbol]['rf_accuracy'] > 0.65:
        return 'random_forest'
    elif historical_performance[symbol]['gb_accuracy'] > 0.60:
        return 'gradient_boost'
    else:
        return 'ensemble'
```

### Cost Management & Monitoring

#### Free Tier Usage Dashboard
```python
def check_free_tier_usage():
    """Monitor free tier resource consumption"""
    usage_report = {
        'compute_hours': get_vm_runtime_hours(),
        'storage_gb': get_storage_usage(),
        'data_transfer_gb': get_transfer_usage(),
        'firestore_reads': get_firestore_reads(),
        'function_invocations': get_function_calls()
    }
    
    # Alert if approaching 80% of any limit
    for resource, usage in usage_report.items():
        if usage > FREE_TIER_LIMITS[resource] * 0.8:
            send_usage_warning(resource, usage)
```

### Implementation Priority

1. **Phase 1**: Core prediction system (current implementation)
2. **Phase 2**: Add Firestore for structured data storage
3. **Phase 3**: Implement BigQuery analytics for performance tracking
4. **Phase 4**: Add Cloud Functions for real-time alerting
5. **Phase 5**: Scale to 50-stock rotation system

This enhanced architecture provides enterprise-grade features while remaining completely within GCP's generous free tier limits. The multi-service approach offers redundancy, better data organization, and advanced analytics capabilities without incurring any costs.