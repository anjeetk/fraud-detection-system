import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from predict import predict_transaction

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sample transaction history
sample_data = pd.DataFrame({
    'ID': [1001, 1002, 1003],
    'Amount': [89.99, 450.00, 1200.00],
    'Card': ['Visa', 'Mastercard', 'Discover'],
    'Status': ['Approved', 'Declined', 'Pending Review']
})

# App layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Real-Time Fraud Detection"), className="mb-4")),  # Fixed: Added missing parenthesis
    
    dbc.Row([
        # Input Column (minimal inputs)
        dbc.Col([
            html.H4("Transaction Details", className="mb-3"),
            dbc.Input(
                id="amount",
                type="number",
                placeholder="Amount ($)",
                min=0,
                step=0.01,
                className="mb-3"
            ),
            dbc.Select(
                id="card-type",
                options=[
                    {"label": "Visa", "value": "visa"},
                    {"label": "Mastercard", "value": "mastercard"},
                    {"label": "American Express", "value": "amex"},
                    {"label": "Discover", "value": "discover"}
                ],
                placeholder="Select card type...",
                className="mb-3"
            ),
            dbc.Button(
                "Analyze Transaction",
                id="analyze-btn",
                color="primary",
                className="w-100"
            ),
            html.Div(id="prediction-output", className="mt-4")
        ], md=5),
        
        # Results Column
        dbc.Col([
            html.H4("Recent Transactions", className="mb-3"),
            dash_table.DataTable(
                id='txn-table',
                columns=[{"name": i, "id": i} for i in sample_data.columns],
                data=sample_data.to_dict('records'),
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Status', 'filter_query': '{Status} = "Declined"'},
                        'backgroundColor': '#FF6B6B',
                        'color': 'white'
                    },
                    {
                        'if': {'column_id': 'Status', 'filter_query': '{Status} = "Pending Review"'},
                        'backgroundColor': '#FFD166',
                        'color': 'black'
                    }
                ]
            ),
            dcc.Graph(
                id='fraud-gauge',
                figure={
                    'data': [{
                        'type': 'indicator',
                        'mode': 'gauge',
                        'value': 0,
                        'gauge': {
                            'axis': {'range': [0, 100]},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "orange"},
                                {'range': [70, 100], 'color': "red"}
                            ]
                        }
                    }],
                    'layout': {'title': 'Fraud Risk Score'}
                }
            )
        ], md=7)
    ])
], fluid=True)

@app.callback(
    [Output('prediction-output', 'children'),
     Output('fraud-gauge', 'figure')],
    Input('analyze-btn', 'n_clicks'),
    [dash.dependencies.State('amount', 'value'),
     dash.dependencies.State('card-type', 'value')]
)
def update_output(n_clicks, amount, card_type):
    if not n_clicks or not amount:
        return [dbc.Alert("Enter amount and card type to analyze", color="info")], dash.no_update
    
    # Build complete transaction with smart defaults
    transaction = {
        # Visible inputs
        'TransactionAmt': float(amount),
        'card4': card_type if card_type else 'visa',
        
        # Hidden defaults (customize for your model)
        'card1': int(np.random.uniform(10000, 99999)),  # Random card number
        'P_emaildomain': 'trusted@bank.com' if amount < 500 else 'new@user.com',
        'dist1': 10 if amount < 1000 else 1000,  # Distance in km
        'DeviceType': 'desktop' if amount < 500 else 'mobile',
        'C1': min(int(amount/100), 20),  # Transaction count estimate
        'D1': 365 if amount < 1000 else 30,  # Card age in days
        'V318': 'V' if amount > 1000 else 'N'  # Verification flag
    }
    
    # Get prediction
    result = predict_transaction(transaction)
    fraud_prob = result['confidence'] * 100
    
    # Risk indicators
    risk_factors = []
    if amount > 1000: risk_factors.append("High amount")
    if card_type in ['discover', 'amex']: risk_factors.append("Rare card type")
    if amount > 500: risk_factors.append("Unusual location pattern")
    if not risk_factors: risk_factors.append("No strong risk factors")
    
    # Alert components
    alert_color = "danger" if fraud_prob > 70 else "warning" if fraud_prob > 30 else "success"
    alert_content = [
        html.H4(f"Fraud Risk: {fraud_prob:.1f}%", className="alert-heading"),
        html.Hr(),
        html.P(f"Amount: ${amount:.2f} | Card: {card_type.upper() if card_type else 'UNSPECIFIED'}"),
        html.P("Key Risk Indicators:"),
        html.Ul([html.Li(factor) for factor in risk_factors])
    ]
    
    # Update gauge
    gauge_fig = {
        'data': [{
            'type': 'indicator',
            'mode': 'gauge',
            'value': fraud_prob,
            'gauge': {'axis': {'range': [0, 100]}}
        }],
        'layout': {'title': 'Live Fraud Risk Score'}
    }
    
    return [dbc.Alert(alert_content, color=alert_color)], gauge_fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)