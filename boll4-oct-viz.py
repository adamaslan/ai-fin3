"""
Technical Analysis Dashboard - Local Server Visualizer
Reads CSV data and creates interactive web dashboard
Run: python visualizer.py
Then open: http://localhost:8000
"""

import pandas as pd
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import threading

class DashboardServer:
    def __init__(self, csv_file, port=8000):
        self.csv_file = csv_file
        self.port = port
        self.data = None
        self.symbol = csv_file.split('_')[0] if '_' in csv_file else 'STOCK'
        
    def load_data(self):
        """Load CSV data"""
        print(f"📂 Loading data from {self.csv_file}...")
        self.data = pd.read_csv(self.csv_file, index_col=0)
        # Try to parse the index as datetime
        try:
            self.data.index = pd.to_datetime(self.data.index)
        except:
            print("⚠️  Warning: Could not parse dates from index")
        print(f"✅ Loaded {len(self.data)} rows of data")
        return self.data
    
    def create_html_dashboard(self):
        """Create interactive HTML dashboard"""
        
        # Prepare data for charts
        df = self.data.tail(200)  # Last 200 days for visualization
        
        # Convert data to JSON
        # Handle different date formats
        if pd.api.types.is_datetime64_any_dtype(df.index):
            dates = df.index.strftime('%Y-%m-%d').tolist()
        else:
            dates = [str(d) for d in df.index.tolist()]
        
        chart_data = {
            'dates': dates,
            'close': df['Close'].fillna(0).tolist(),
            'volume': df['Volume'].fillna(0).tolist(),
            'sma_10': df['SMA_10'].fillna(0).tolist(),
            'sma_20': df['SMA_20'].fillna(0).tolist(),
            'sma_50': df['SMA_50'].fillna(0).tolist(),
            'sma_200': df['SMA_200'].fillna(0).tolist(),
            'ema_10': df['EMA_10'].fillna(0).tolist(),
            'ema_20': df['EMA_20'].fillna(0).tolist(),
            'rsi': df['RSI'].fillna(0).tolist(),
            'macd': df['MACD'].fillna(0).tolist(),
            'macd_signal': df['MACD_Signal'].fillna(0).tolist(),
            'macd_hist': df['MACD_Hist'].fillna(0).tolist(),
            'bb_upper': df['BB_Upper'].fillna(0).tolist(),
            'bb_middle': df['BB_Middle'].fillna(0).tolist(),
            'bb_lower': df['BB_Lower'].fillna(0).tolist(),
            'stoch_k': df['Stoch_K'].fillna(0).tolist(),
            'stoch_d': df['Stoch_D'].fillna(0).tolist(),
            'atr': df['ATR'].fillna(0).tolist(),
            'adx': df['ADX'].fillna(0).tolist(),
            'cci': df['CCI'].fillna(0).tolist(),
            'williams_r': df['Williams_R'].fillna(0).tolist(),
            'mfi': df['MFI'].fillna(0).tolist(),
            'obv': df['OBV'].fillna(0).tolist(),
            'volatility': df['Volatility'].fillna(0).tolist(),
        }
        
        # Get current values
        current = df.iloc[-1]
        
        # Format date safely
        if pd.api.types.is_datetime64_any_dtype(df.index):
            current_date = current.name.strftime('%Y-%m-%d')
        else:
            current_date = str(current.name)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Technical Analysis Dashboard - {self.symbol}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .metric-card .label {{
            color: #888;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        
        .metric-card.bullish .value {{
            color: #10b981;
        }}
        
        .metric-card.bearish .value {{
            color: #ef4444;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }}
        
        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        
        .chart-container h2 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        
        .chart-wrapper {{
            position: relative;
            height: 400px;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .tab {{
            padding: 10px 20px;
            background: #f0f0f0;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }}
        
        .tab:hover {{
            background: #e0e0e0;
        }}
        
        .tab.active {{
            background: #667eea;
            color: white;
        }}
        
        .chart-section {{
            display: none;
        }}
        
        .chart-section.active {{
            display: block;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {self.symbol} Technical Analysis Dashboard</h1>
            <p class="subtitle">Real-time visualization of 20 technical indicators</p>
            <p class="subtitle">Last Updated: {current_date}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Current Price</div>
                <div class="value">${current['Close']:.2f}</div>
            </div>
            <div class="metric-card {'bullish' if current['RSI'] < 50 else 'bearish'}">
                <div class="label">RSI (14)</div>
                <div class="value">{current['RSI']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">MACD</div>
                <div class="value">{current['MACD']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="label">ADX</div>
                <div class="value">{current['ADX']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Volatility</div>
                <div class="value">{current['Volatility']:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Volume</div>
                <div class="value">{current['Volume']/1000000:.1f}M</div>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showChart('price')">Price & MAs</button>
            <button class="tab" onclick="showChart('bollinger')">Bollinger Bands</button>
            <button class="tab" onclick="showChart('rsi')">RSI</button>
            <button class="tab" onclick="showChart('macd')">MACD</button>
            <button class="tab" onclick="showChart('stochastic')">Stochastic</button>
            <button class="tab" onclick="showChart('volume')">Volume & OBV</button>
            <button class="tab" onclick="showChart('oscillators')">Oscillators</button>
            <button class="tab" onclick="showChart('trend')">Trend Indicators</button>
        </div>
        
        <div id="price" class="chart-section active">
            <div class="chart-container">
                <h2>Price Action & Moving Averages</h2>
                <div class="chart-wrapper">
                    <canvas id="priceChart"></canvas>
                </div>
            </div>
        </div>
        
        <div id="bollinger" class="chart-section">
            <div class="chart-container">
                <h2>Bollinger Bands</h2>
                <div class="chart-wrapper">
                    <canvas id="bollingerChart"></canvas>
                </div>
            </div>
        </div>
        
        <div id="rsi" class="chart-section">
            <div class="chart-container">
                <h2>Relative Strength Index (RSI)</h2>
                <div class="chart-wrapper">
                    <canvas id="rsiChart"></canvas>
                </div>
            </div>
        </div>
        
        <div id="macd" class="chart-section">
            <div class="chart-container">
                <h2>MACD (Moving Average Convergence Divergence)</h2>
                <div class="chart-wrapper">
                    <canvas id="macdChart"></canvas>
                </div>
            </div>
        </div>
        
        <div id="stochastic" class="chart-section">
            <div class="chart-container">
                <h2>Stochastic Oscillator</h2>
                <div class="chart-wrapper">
                    <canvas id="stochasticChart"></canvas>
                </div>
            </div>
        </div>
        
        <div id="volume" class="chart-section">
            <div class="chart-container">
                <h2>Volume & On-Balance Volume</h2>
                <div class="chart-wrapper">
                    <canvas id="volumeChart"></canvas>
                </div>
            </div>
        </div>
        
        <div id="oscillators" class="chart-section">
            <div class="chart-container">
                <h2>Technical Oscillators (CCI, Williams %R, MFI)</h2>
                <div class="chart-wrapper">
                    <canvas id="oscillatorsChart"></canvas>
                </div>
            </div>
        </div>
        
        <div id="trend" class="chart-section">
            <div class="chart-container">
                <h2>Trend Indicators (ADX, ATR, Volatility)</h2>
                <div class="chart-wrapper">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const chartData = {json.dumps(chart_data)};
        
        const commonOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    position: 'top',
                }},
            }},
            interaction: {{
                intersect: false,
                mode: 'index',
            }},
        }};
        
        // Price Chart
        new Chart(document.getElementById('priceChart'), {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [
                    {{
                        label: 'Close Price',
                        data: chartData.close,
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                    }},
                    {{
                        label: '10 EMA',
                        data: chartData.ema_10,
                        borderColor: 'rgb(34, 197, 94)',
                        borderWidth: 2,
                        fill: false,
                    }},
                    {{
                        label: '20 EMA',
                        data: chartData.ema_20,
                        borderColor: 'rgb(251, 146, 60)',
                        borderWidth: 2,
                        fill: false,
                    }},
                    {{
                        label: '50 SMA',
                        data: chartData.sma_50,
                        borderColor: 'rgb(236, 72, 153)',
                        borderWidth: 2,
                        fill: false,
                    }},
                    {{
                        label: '200 SMA',
                        data: chartData.sma_200,
                        borderColor: 'rgb(139, 92, 246)',
                        borderWidth: 2,
                        fill: false,
                    }},
                ]
            }},
            options: commonOptions
        }});
        
        // Bollinger Bands Chart
        new Chart(document.getElementById('bollingerChart'), {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [
                    {{
                        label: 'Close Price',
                        data: chartData.close,
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        fill: false,
                    }},
                    {{
                        label: 'Upper Band',
                        data: chartData.bb_upper,
                        borderColor: 'rgb(239, 68, 68)',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        fill: false,
                    }},
                    {{
                        label: 'Middle Band',
                        data: chartData.bb_middle,
                        borderColor: 'rgb(107, 114, 128)',
                        borderWidth: 1,
                        fill: false,
                    }},
                    {{
                        label: 'Lower Band',
                        data: chartData.bb_lower,
                        borderColor: 'rgb(34, 197, 94)',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        fill: false,
                    }},
                ]
            }},
            options: commonOptions
        }});
        
        // RSI Chart
        new Chart(document.getElementById('rsiChart'), {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [
                    {{
                        label: 'RSI',
                        data: chartData.rsi,
                        borderColor: 'rgb(139, 92, 246)',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                    }},
                ]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    annotation: {{
                        annotations: {{
                            overbought: {{
                                type: 'line',
                                yMin: 70,
                                yMax: 70,
                                borderColor: 'rgb(239, 68, 68)',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {{
                                    content: 'Overbought (70)',
                                    enabled: true,
                                }}
                            }},
                            oversold: {{
                                type: 'line',
                                yMin: 30,
                                yMax: 30,
                                borderColor: 'rgb(34, 197, 94)',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {{
                                    content: 'Oversold (30)',
                                    enabled: true,
                                }}
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
        
        // MACD Chart
        new Chart(document.getElementById('macdChart'), {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [
                    {{
                        label: 'MACD',
                        data: chartData.macd,
                        borderColor: 'rgb(59, 130, 246)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y',
                    }},
                    {{
                        label: 'Signal',
                        data: chartData.macd_signal,
                        borderColor: 'rgb(239, 68, 68)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y',
                    }},
                    {{
                        label: 'Histogram',
                        data: chartData.macd_hist,
                        type: 'bar',
                        backgroundColor: 'rgba(34, 197, 94, 0.5)',
                        yAxisID: 'y',
                    }},
                ]
            }},
            options: {{
                ...commonOptions,
                scales: {{
                    y: {{
                        type: 'linear',
                        position: 'left',
                    }}
                }}
            }}
        }});
        
        // Stochastic Chart
        new Chart(document.getElementById('stochasticChart'), {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [
                    {{
                        label: '%K',
                        data: chartData.stoch_k,
                        borderColor: 'rgb(59, 130, 246)',
                        borderWidth: 2,
                        fill: false,
                    }},
                    {{
                        label: '%D',
                        data: chartData.stoch_d,
                        borderColor: 'rgb(239, 68, 68)',
                        borderWidth: 2,
                        fill: false,
                    }},
                ]
            }},
            options: {{
                ...commonOptions,
                scales: {{
                    y: {{
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
        
        // Volume Chart
        new Chart(document.getElementById('volumeChart'), {{
            type: 'bar',
            data: {{
                labels: chartData.dates,
                datasets: [
                    {{
                        label: 'Volume',
                        data: chartData.volume,
                        backgroundColor: 'rgba(59, 130, 246, 0.5)',
                        yAxisID: 'y',
                    }},
                    {{
                        label: 'OBV',
                        data: chartData.obv,
                        type: 'line',
                        borderColor: 'rgb(239, 68, 68)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1',
                    }},
                ]
            }},
            options: {{
                ...commonOptions,
                scales: {{
                    y: {{
                        type: 'linear',
                        position: 'left',
                    }},
                    y1: {{
                        type: 'linear',
                        position: 'right',
                        grid: {{
                            drawOnChartArea: false,
                        }},
                    }}
                }}
            }}
        }});
        
        // Oscillators Chart
        new Chart(document.getElementById('oscillatorsChart'), {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [
                    {{
                        label: 'CCI',
                        data: chartData.cci,
                        borderColor: 'rgb(59, 130, 246)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y',
                    }},
                    {{
                        label: 'Williams %R',
                        data: chartData.williams_r,
                        borderColor: 'rgb(239, 68, 68)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1',
                    }},
                    {{
                        label: 'MFI',
                        data: chartData.mfi,
                        borderColor: 'rgb(34, 197, 94)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y2',
                    }},
                ]
            }},
            options: {{
                ...commonOptions,
                scales: {{
                    y: {{
                        type: 'linear',
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'CCI'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        position: 'right',
                        grid: {{
                            drawOnChartArea: false,
                        }},
                        title: {{
                            display: true,
                            text: 'Williams %R'
                        }}
                    }},
                    y2: {{
                        type: 'linear',
                        position: 'right',
                        grid: {{
                            drawOnChartArea: false,
                        }},
                        title: {{
                            display: true,
                            text: 'MFI'
                        }}
                    }}
                }}
            }}
        }});
        
        // Trend Indicators Chart
        new Chart(document.getElementById('trendChart'), {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [
                    {{
                        label: 'ADX',
                        data: chartData.adx,
                        borderColor: 'rgb(59, 130, 246)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y',
                    }},
                    {{
                        label: 'ATR',
                        data: chartData.atr,
                        borderColor: 'rgb(239, 68, 68)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1',
                    }},
                    {{
                        label: 'Volatility %',
                        data: chartData.volatility,
                        borderColor: 'rgb(34, 197, 94)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y',
                    }},
                ]
            }},
            options: {{
                ...commonOptions,
                scales: {{
                    y: {{
                        type: 'linear',
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'ADX / Volatility'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        position: 'right',
                        grid: {{
                            drawOnChartArea: false,
                        }},
                        title: {{
                            display: true,
                            text: 'ATR'
                        }}
                    }}
                }}
            }}
        }});
        
        function showChart(chartName) {{
            // Hide all sections
            document.querySelectorAll('.chart-section').forEach(section => {{
                section.classList.remove('active');
            }});
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected section
            document.getElementById(chartName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
        """
        
        # Save HTML file
        html_file = 'dashboard.html'
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Dashboard created: {html_file}")
        return html_file
    
    def start_server(self):
        """Start local web server"""
        
        class CustomHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress server logs
        
        server_address = ('', self.port)
        httpd = HTTPServer(server_address, CustomHandler)
        
        print(f"\n🌐 Starting server at http://localhost:{self.port}")
        print(f"📊 Dashboard URL: http://localhost:{self.port}/dashboard.html")
        print("\n✅ Server is running... Press Ctrl+C to stop\n")
        
        # Open browser
        webbrowser.open(f'http://localhost:{self.port}/dashboard.html')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped")
            httpd.shutdown()

def main():
    """Main execution"""
    import sys
    
    print("=" * 80)
    print("📊 TECHNICAL ANALYSIS DASHBOARD VISUALIZER")
    print("=" * 80)
    
    # Default CSV file - CHANGE THIS TO YOUR FILE
    default_csv = "RGTI_technical_analysis.csv"
    
    # Get CSV file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print(f"\n💡 Default file: {default_csv}")
        user_input = input(f"📂 Press Enter to use default, or type CSV filename: ").strip()
        csv_file = user_input if user_input else default_csv
    
    if not os.path.exists(csv_file):
        print(f"\n❌ Error: File '{csv_file}' not found")
        print("\nAvailable CSV files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"   • {file}")
        return
    
    try:
        # Create dashboard
        dashboard = DashboardServer(csv_file)
        dashboard.load_data()
        dashboard.create_html_dashboard()
        
        # Start server
        dashboard.start_server()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()