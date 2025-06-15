import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ThreatDetectionVisualizer:
    def __init__(self):
        self.rf_model = None
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = {}
        self.anomaly_scores = None
        
    def load_and_preprocess_data(self, data_path=None, dataset_type='kdd99'):
        """Load and preprocess the dataset"""
        if dataset_type == 'kdd99':
            # KDD99 column names
            columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                      'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                      'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                      'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                      'num_access_files', 'num_outbound_cmds', 'is_host_login',
                      'is_guest_login', 'count', 'srv_count', 'serror_rate',
                      'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                      'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                      'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                      'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                      'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                      'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                      'dst_host_srv_rerror_rate', 'label']
            
            # Generate sample data for demonstration
            np.random.seed(42)
            n_samples = 10000
            
            # Generate synthetic KDD99-like data
            data = {}
            for col in columns[:-1]:  # All except label
                if col in ['protocol_type', 'service', 'flag']:
                    data[col] = np.random.choice(['tcp', 'udp', 'icmp'], n_samples)
                elif col in ['land', 'logged_in', 'is_host_login', 'is_guest_login']:
                    data[col] = np.random.choice([0, 1], n_samples)
                else:
                    data[col] = np.random.exponential(2, n_samples)
            
            # Generate labels (normal vs attack)
            attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
            data['label'] = np.random.choice(attack_types, n_samples, 
                                           p=[0.6, 0.2, 0.1, 0.05, 0.05])
            
            df = pd.DataFrame(data)
            
        else:  # CICIDS2017
            # Generate synthetic CICIDS2017-like data
            np.random.seed(42)
            n_samples = 10000
            
            columns = ['flow_duration', 'total_fwd_packets', 'total_backward_packets',
                      'total_length_fwd_packets', 'total_length_bwd_packets',
                      'fwd_packet_length_max', 'fwd_packet_length_min',
                      'fwd_packet_length_mean', 'fwd_packet_length_std',
                      'bwd_packet_length_max', 'bwd_packet_length_min',
                      'bwd_packet_length_mean', 'bwd_packet_length_std',
                      'flow_bytes_s', 'flow_packets_s', 'flow_iat_mean',
                      'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
                      'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std',
                      'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_total',
                      'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',
                      'label']
            
            data = {}
            for col in columns[:-1]:
                data[col] = np.random.exponential(5, n_samples)
            
            attack_types = ['BENIGN', 'DoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack', 'Brute Force']
            data['label'] = np.random.choice(attack_types, n_samples,
                                           p=[0.7, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02])
            
            df = pd.DataFrame(data)
        
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for ML models"""
        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Binary classification: normal (0) vs attack (1)
        y_binary = (y != 'normal') & (y != 'BENIGN')
        y_binary = y_binary.astype(int)
        
        return X, y, y_binary
    
    def train_models(self, X, y_binary):
        """Train Random Forest and Isolation Forest models"""
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(self.X_train_scaled)
        
        # Make predictions
        self.predictions['rf'] = self.rf_model.predict(self.X_test_scaled)
        self.predictions['rf_proba'] = self.rf_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Isolation Forest predictions (-1 for outliers, 1 for inliers)
        if_pred = self.isolation_forest.predict(self.X_test_scaled)
        self.predictions['if'] = (if_pred == -1).astype(int)  # Convert to 0/1
        self.anomaly_scores = self.isolation_forest.score_samples(self.X_test_scaled)
    
    def create_risk_dashboard(self):
        """Create comprehensive risk visualization dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Threat Risk Level Distribution', 'Attack Detection Over Time',
                          'Model Performance Comparison', 'Anomaly Score Distribution',
                          'Feature Importance (Top 10)', 'Real-time Threat Status'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # 1. Risk Level Pie Chart
        risk_levels = self._calculate_risk_levels()
        fig.add_trace(
            go.Pie(labels=list(risk_levels.keys()), 
                  values=list(risk_levels.values()),
                  hole=0.4,
                  marker_colors=['green', 'yellow', 'orange', 'red']),
            row=1, col=1
        )
        
        # 2. Attack Detection Timeline
        timeline_data = self._generate_timeline_data()
        fig.add_trace(
            go.Scatter(x=timeline_data['time'], 
                      y=timeline_data['attacks'],
                      mode='lines+markers',
                      name='Detected Attacks',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        # 3. Model Performance Comparison
        performance_metrics = self._calculate_performance_metrics()
        models = list(performance_metrics.keys())
        precision = [performance_metrics[m]['precision'] for m in models]
        recall = [performance_metrics[m]['recall'] for m in models]
        f1 = [performance_metrics[m]['f1'] for m in models]
        
        fig.add_trace(go.Bar(x=models, y=precision, name='Precision', marker_color='lightblue'), row=2, col=1)
        fig.add_trace(go.Bar(x=models, y=recall, name='Recall', marker_color='lightgreen'), row=2, col=1)
        fig.add_trace(go.Bar(x=models, y=f1, name='F1-Score', marker_color='lightyellow'), row=2, col=1)
        
        # 4. Anomaly Score Distribution
        fig.add_trace(
            go.Histogram(x=self.anomaly_scores, 
                        nbinsx=50,
                        name='Anomaly Scores',
                        marker_color='purple',
                        opacity=0.7),
            row=2, col=2
        )
        
        # 5. Feature Importance
        if hasattr(self.rf_model, 'feature_importances_'):
            feature_names = [f'Feature_{i}' for i in range(len(self.rf_model.feature_importances_))]
            importances = self.rf_model.feature_importances_
            top_features = sorted(zip(feature_names, importances), 
                                key=lambda x: x[1], reverse=True)[:10]
            
            fig.add_trace(
                go.Bar(x=[f[1] for f in top_features],
                      y=[f[0] for f in top_features],
                      orientation='h',
                      marker_color='orange'),
                row=3, col=1
            )
        
        # 6. Real-time Threat Status Gauge
        current_threat_level = np.mean(self.predictions['rf_proba']) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_threat_level,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current Threat Level (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Cybersecurity Threat Detection Dashboard",
            title_x=0.5,
            showlegend=True,
            font=dict(size=10)
        )
        
        return fig
    
    def create_detailed_analysis_charts(self):
        """Create detailed analysis charts"""
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.predictions['rf_proba'])
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, 
                                    mode='lines',
                                    name=f'ROC Curve (AUC = {roc_auc:.2f})',
                                    line=dict(color='blue', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                    mode='lines',
                                    name='Random Classifier',
                                    line=dict(color='red', dash='dash')))
        fig_roc.update_layout(
            title='ROC Curve - Random Forest Classifier',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500
        )
        
        # Confusion Matrix Heatmap
        cm = confusion_matrix(self.y_test, self.predictions['rf'])
        fig_cm = go.Figure(data=go.Heatmap(z=cm,
                                          x=['Normal', 'Attack'],
                                          y=['Normal', 'Attack'],
                                          colorscale='Blues',
                                          text=cm,
                                          texttemplate="%{text}",
                                          textfont={"size": 16},
                                          hoverongaps=False))
        fig_cm.update_layout(
            title='Confusion Matrix - Random Forest',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        
        return fig_roc, fig_cm
    
    def _calculate_risk_levels(self):
        """Calculate risk level distribution"""
        proba = self.predictions['rf_proba']
        risk_levels = {
            'Low (0-25%)': np.sum((proba >= 0) & (proba < 0.25)),
            'Medium (25-50%)': np.sum((proba >= 0.25) & (proba < 0.5)),
            'High (50-75%)': np.sum((proba >= 0.5) & (proba < 0.75)),
            'Critical (75-100%)': np.sum(proba >= 0.75)
        }
        return risk_levels
    
    def _generate_timeline_data(self):
        """Generate timeline data for attack detection"""
        # Simulate time-based attack detection
        time_points = pd.date_range(start='2024-01-01', periods=24, freq='H')
        attacks = np.random.poisson(3, 24)  # Simulate attack counts
        return {'time': time_points, 'attacks': attacks}
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics for both models"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {}
        
        # Random Forest metrics
        rf_precision = precision_score(self.y_test, self.predictions['rf'])
        rf_recall = recall_score(self.y_test, self.predictions['rf'])
        rf_f1 = f1_score(self.y_test, self.predictions['rf'])
        
        metrics['Random Forest'] = {
            'precision': rf_precision,
            'recall': rf_recall,
            'f1': rf_f1
        }
        
        # Isolation Forest metrics
        if_precision = precision_score(self.y_test, self.predictions['if'])
        if_recall = recall_score(self.y_test, self.predictions['if'])
        if_f1 = f1_score(self.y_test, self.predictions['if'])
        
        metrics['Isolation Forest'] = {
            'precision': if_precision,
            'recall': if_recall,
            'f1': if_f1
        }
        
        return metrics
    
    def generate_threat_report(self):
        """Generate a comprehensive threat analysis report"""
        total_samples = len(self.y_test)
        total_attacks = np.sum(self.y_test)
        detection_rate = np.mean(self.predictions['rf'] == self.y_test) * 100
        
        report = f"""
        CYBERSECURITY THREAT DETECTION REPORT
        ====================================
        
        Dataset Analysis:
        - Total Samples Analyzed: {total_samples:,}
        - Actual Attacks Detected: {total_attacks:,}
        - Attack Rate: {(total_attacks/total_samples)*100:.2f}%
        
        Model Performance:
        - Overall Detection Accuracy: {detection_rate:.2f}%
        - Random Forest Precision: {self._calculate_performance_metrics()['Random Forest']['precision']:.3f}
        - Random Forest Recall: {self._calculate_performance_metrics()['Random Forest']['recall']:.3f}
        - Random Forest F1-Score: {self._calculate_performance_metrics()['Random Forest']['f1']:.3f}
        
        Risk Assessment:
        {self._format_risk_levels()}
        
        Recommendations:
        - Monitor high-risk connections closely
        - Implement additional security measures for critical threats
        - Regular model retraining recommended
        - Consider ensemble methods for improved accuracy
        """
        
        return report
    
    def _format_risk_levels(self):
        """Format risk levels for report"""
        risk_levels = self._calculate_risk_levels()
        formatted = ""
        for level, count in risk_levels.items():
            percentage = (count / len(self.predictions['rf_proba'])) * 100
            formatted += f"- {level}: {count} samples ({percentage:.1f}%)\n"
        return formatted

# Main execution
def main():
    """Main function to run the threat detection visualization"""
    print("Initializing Threat Detection Visualization System...")
    
    # Initialize visualizer
    viz = ThreatDetectionVisualizer()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = viz.load_and_preprocess_data(dataset_type='kdd99')  # Change to 'cicids2017' if needed
    X, y, y_binary = viz.preprocess_features(df)
    
    # Train models
    print("Training machine learning models...")
    viz.train_models(X, y_binary)
    
    # Create visualizations
    print("Generating visualizations...")
    
    # Main dashboard
    dashboard = viz.create_risk_dashboard()
    dashboard.show()
    
    # Detailed analysis charts
    roc_chart, confusion_matrix = viz.create_detailed_analysis_charts()
    roc_chart.show()
    confusion_matrix.show()
    
    # Generate report
    report = viz.generate_threat_report()
    print("\n" + report)
    
    # Save visualizations (optional)
    dashboard.write_html("threat_detection_dashboard.html")
    roc_chart.write_html("roc_curve_analysis.html")
    confusion_matrix.write_html("confusion_matrix.html")
    
    print("\nVisualization files saved:")
    print("- threat_detection_dashboard.html")
    print("- roc_curve_analysis.html") 
    print("- confusion_matrix.html")

if __name__ == "__main__":
    main()