#!/usr/bin/env python3
"""
AI-Powered Threat Detection with Real Cybersecurity Datasets
Enhanced version using KDD Cup 1999 and CICIDS2017 datasets
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import urllib.request
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetManager:
    """Manage downloading and loading of cybersecurity datasets"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.create_data_directory()
        
        # Dataset configurations
        self.datasets = {
            'kdd99': {
                'url': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz',
                'filename': 'kddcup99_10percent.gz',
                'columns': [
                    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
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
                    'dst_host_srv_rerror_rate', 'label'
                ]
            }
        }
    
    def create_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def download_kdd99(self) -> str:
        """Download KDD Cup 1999 dataset"""
        dataset_info = self.datasets['kdd99']
        filepath = os.path.join(self.data_dir, dataset_info['filename'])
        
        if os.path.exists(filepath):
            logger.info("KDD99 dataset already exists")
            return filepath
        
        try:
            logger.info("Downloading KDD Cup 1999 dataset...")
            urllib.request.urlretrieve(dataset_info['url'], filepath)
            logger.info(f"Downloaded KDD99 dataset to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to download KDD99 dataset: {e}")
            return None
    
    def load_kdd99(self) -> pd.DataFrame:
        """Load and preprocess KDD Cup 1999 dataset"""
        filepath = self.download_kdd99()
        if not filepath:
            return None
        
        try:
            logger.info("Loading KDD99 dataset...")
            df = pd.read_csv(filepath, names=self.datasets['kdd99']['columns'])
            
            # Basic preprocessing
            logger.info(f"Loaded {len(df)} records from KDD99 dataset")
            logger.info(f"Attack distribution: {df['label'].value_counts().head()}")
            
            return df
        except Exception as e:
            logger.error(f"Failed to load KDD99 dataset: {e}")
            return None
    
    def create_sample_cicids2017(self) -> pd.DataFrame:
        """Create a sample CICIDS2017-like dataset for demonstration"""
        logger.info("Creating sample CICIDS2017-like dataset...")
        
        np.random.seed(42)
        n_samples = 10000
        
        # Simulate realistic network flow features
        data = {
            'Flow_Duration': np.random.exponential(1000, n_samples),
            'Total_Fwd_Packets': np.random.poisson(50, n_samples),
            'Total_Backward_Packets': np.random.poisson(30, n_samples),
            'Total_Length_of_Fwd_Packets': np.random.exponential(2000, n_samples),
            'Total_Length_of_Bwd_Packets': np.random.exponential(1500, n_samples),
            'Fwd_Packet_Length_Max': np.random.exponential(500, n_samples),
            'Bwd_Packet_Length_Max': np.random.exponential(400, n_samples),
            'Flow_Bytes_per_sec': np.random.exponential(10000, n_samples),
            'Flow_Packets_per_sec': np.random.exponential(100, n_samples),
            'Flow_IAT_Mean': np.random.exponential(100, n_samples),
            'Fwd_IAT_Mean': np.random.exponential(80, n_samples),
            'Bwd_IAT_Mean': np.random.exponential(120, n_samples),
            'Active_Mean': np.random.exponential(200, n_samples),
            'Idle_Mean': np.random.exponential(500, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic labels (90% benign, 10% attacks)
        attack_types = ['BENIGN', 'DoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack', 'Brute Force']
        weights = [0.85, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01]
        df['Label'] = np.random.choice(attack_types, n_samples, p=weights)
        
        # Add some realistic anomalies for attacks
        attack_mask = df['Label'] != 'BENIGN'
        df.loc[attack_mask, 'Flow_Bytes_per_sec'] *= np.random.uniform(2, 10, attack_mask.sum())
        df.loc[attack_mask, 'Flow_Packets_per_sec'] *= np.random.uniform(1.5, 5, attack_mask.sum())
        
        logger.info(f"Created sample dataset with {len(df)} records")
        logger.info(f"Label distribution:\n{df['Label'].value_counts()}")
        
        return df

class AdvancedThreatDetector:
    """Enhanced threat detection using multiple ML algorithms"""
    
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.scalers = {}
        self.encoders = {}
        self.is_trained = False
        self.feature_columns = []
        
    def preprocess_kdd99(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess KDD99 dataset for ML"""
        logger.info("Preprocessing KDD99 dataset...")
        
        # Separate features and labels
        X = df.drop(['label'], axis=1)
        y = df['label']
        
        # Create binary labels (normal vs attack)
        y_binary = (y != 'normal.').astype(int)
        
        # Handle categorical features
        categorical_features = ['protocol_type', 'service', 'flag']
        
        for feature in categorical_features:
            if feature not in self.encoders:
                self.encoders[feature] = LabelEncoder()
                X[feature] = self.encoders[feature].fit_transform(X[feature].astype(str))
            else:
                X[feature] = self.encoders[feature].transform(X[feature].astype(str))
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale numerical features
        if 'kdd99' not in self.scalers:
            self.scalers['kdd99'] = StandardScaler()
            X_scaled = self.scalers['kdd99'].fit_transform(X)
        else:
            X_scaled = self.scalers['kdd99'].transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        return X_scaled, y_binary
    
    def preprocess_cicids(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess CICIDS2017 dataset for ML"""
        logger.info("Preprocessing CICIDS2017 dataset...")
        
        # Separate features and labels
        X = df.drop(['Label'], axis=1)
        y = df['Label']
        
        # Create binary labels (benign vs attack)
        y_binary = (y != 'BENIGN').astype(int)
        
        # Handle any potential infinity or NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        if 'cicids' not in self.scalers:
            self.scalers['cicids'] = StandardScaler()
            X_scaled = self.scalers['cicids'].fit_transform(X)
        else:
            X_scaled = self.scalers['cicids'].transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        return X_scaled, y_binary
    
    def train(self, X: pd.DataFrame, y: pd.Series, dataset_type: str = 'kdd99'):
        """Train multiple models on the dataset"""
        logger.info(f"Training models on {dataset_type} dataset...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # Train Isolation Forest (unsupervised)
        logger.info("Training Isolation Forest...")
        self.models['isolation_forest'].fit(X_train)
        
        # Predict and evaluate
        y_pred_iso = self.models['isolation_forest'].predict(X_test)
        y_pred_iso = (y_pred_iso == -1).astype(int)  # Convert to binary
        
        results['isolation_forest'] = {
            'accuracy': accuracy_score(y_test, y_pred_iso),
            'classification_report': classification_report(y_test, y_pred_iso, output_dict=True)
        }
        
        # Train Random Forest (supervised)
        logger.info("Training Random Forest...")
        self.models['random_forest'].fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred_rf = self.models['random_forest'].predict(X_test)
        
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'classification_report': classification_report(y_test, y_pred_rf, output_dict=True)
        }
        
        self.is_trained = True
        
        # Log results
        logger.info("=== MODEL PERFORMANCE ===")
        for model_name, metrics in results.items():
            logger.info(f"{model_name.upper()}:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['classification_report']['1']['precision']:.4f}")
            logger.info(f"  Recall: {metrics['classification_report']['1']['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['classification_report']['1']['f1-score']:.4f}")
        
        return results
    
    def detect_threats_realtime(self, X: pd.DataFrame) -> List[Dict]:
        """Detect threats in real-time data"""
        if not self.is_trained:
            raise ValueError("Models must be trained before detection")
        
        threats = []
        
        # Get predictions from both models
        iso_predictions = self.models['isolation_forest'].predict(X)
        rf_predictions = self.models['random_forest'].predict(X)
        
        # Get anomaly scores
        iso_scores = self.models['isolation_forest'].decision_function(X)
        rf_probabilities = self.models['random_forest'].predict_proba(X)[:, 1]
        
        for i in range(len(X)):
            # Ensemble decision: threat if either model detects anomaly
            is_threat_iso = iso_predictions[i] == -1
            is_threat_rf = rf_predictions[i] == 1
            
            if is_threat_iso or is_threat_rf:
                threat = {
                    'index': i,
                    'timestamp': datetime.now().isoformat(),
                    'isolation_forest_score': float(iso_scores[i]),
                    'random_forest_probability': float(rf_probabilities[i]),
                    'detected_by': [],
                    'risk_level': self._calculate_ensemble_risk(iso_scores[i], rf_probabilities[i]),
                    'features': X.iloc[i].to_dict()
                }
                
                if is_threat_iso:
                    threat['detected_by'].append('Isolation Forest')
                if is_threat_rf:
                    threat['detected_by'].append('Random Forest')
                
                threats.append(threat)
        
        return threats
    
    def _calculate_ensemble_risk(self, iso_score: float, rf_prob: float) -> str:
        """Calculate risk level using ensemble scoring"""
        # Normalize scores
        iso_risk = max(0, min(1, (-iso_score + 0.5) * 2))  # Convert to 0-1 scale
        rf_risk = rf_prob
        
        # Ensemble score (weighted average)
        ensemble_score = 0.6 * rf_risk + 0.4 * iso_risk
        
        if ensemble_score > 0.8:
            return "CRITICAL"
        elif ensemble_score > 0.6:
            return "HIGH"
        elif ensemble_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

class EnhancedIncidentResponder:
    """Enhanced incident response with dataset-specific actions"""
    
    def __init__(self):
        self.response_playbooks = {
            'DoS': ['rate_limit', 'block_source', 'scale_resources'],
            'PortScan': ['block_source', 'enhance_monitoring', 'log_detailed'],
            'Bot': ['isolate_host', 'deep_scan', 'block_c2'],
            'Web Attack': ['block_source', 'patch_vulnerability', 'enhance_waf'],
            'Brute Force': ['account_lockout', 'block_source', 'mfa_enforce'],
            'Infiltration': ['isolate_network', 'forensic_capture', 'incident_escalate']
        }
    
    def respond_to_dataset_threat(self, threat: Dict, dataset_type: str) -> Dict:
        """Respond to threats with dataset-specific context"""
        response = {
            'threat_id': f"THR_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'timestamp': datetime.now().isoformat(),
            'dataset_type': dataset_type,
            'risk_level': threat['risk_level'],
            'detected_by': threat['detected_by'],
            'actions_taken': [],
            'status': 'SUCCESS'
        }
        
        # Determine attack type based on feature analysis
        attack_type = self._classify_attack_type(threat['features'], dataset_type)
        response['attack_type'] = attack_type
        
        # Get appropriate playbook
        actions = self.response_playbooks.get(attack_type, ['monitor', 'log_incident'])
        
        for action in actions:
            try:
                result = self._execute_enhanced_action(action, threat, dataset_type)
                response['actions_taken'].append({'action': action, 'result': result})
            except Exception as e:
                logger.error(f"Failed to execute action {action}: {e}")
                response['status'] = 'PARTIAL_FAILURE'
        
        return response
    
    def _classify_attack_type(self, features: Dict, dataset_type: str) -> str:
        """Classify attack type based on features"""
        if dataset_type == 'kdd99':
            # KDD99 specific classification logic
            if features.get('src_bytes', 0) > 10000 or features.get('dst_bytes', 0) > 10000:
                return 'DoS'
            elif features.get('count', 0) > 100:
                return 'PortScan'
            else:
                return 'Unknown'
        
        elif dataset_type == 'cicids':
            # CICIDS2017 specific classification logic
            if features.get('Flow_Packets_per_sec', 0) > 1000:
                return 'DoS'
            elif features.get('Flow_Duration', 0) < 100:
                return 'PortScan'
            else:
                return 'Unknown'
        
        return 'Unknown'
    
    def _execute_enhanced_action(self, action: str, threat: Dict, dataset_type: str) -> str:
        """Execute enhanced response actions"""
        logger.warning(f"EXECUTING {action.upper()} for {dataset_type} threat")
        
        action_results = {
            'rate_limit': f"Applied rate limiting based on {dataset_type} metrics",
            'block_source': f"Blocked source IP/host identified in {dataset_type} features",
            'scale_resources': f"Auto-scaled resources to handle {dataset_type} attack pattern",
            'enhance_monitoring': f"Enhanced monitoring for {dataset_type} attack signatures",
            'isolate_host': f"Isolated compromised host based on {dataset_type} analysis",
            'forensic_capture': f"Captured forensic data using {dataset_type} context"
        }
        
        return action_results.get(action, f"Executed {action} with {dataset_type} context")

# Main execution class
class RealDatasetThreatSystem:
    """Complete threat detection system using real cybersecurity datasets"""
    
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.detector = AdvancedThreatDetector()
        self.responder = EnhancedIncidentResponder()
        self.incident_log = []
    
    def run_kdd99_analysis(self):
        """Run complete analysis on KDD Cup 1999 dataset"""
        logger.info("=== KDD CUP 1999 ANALYSIS ===")
        
        # Load dataset
        df = self.dataset_manager.load_kdd99()
        if df is None:
            logger.error("Failed to load KDD99 dataset")
            return
        
        # Sample for demo (use full dataset for production)
        df_sample = df.sample(n=5000, random_state=42)
        logger.info(f"Using sample of {len(df_sample)} records for demo")
        
        # Preprocess
        X, y = self.detector.preprocess_kdd99(df_sample)
        
        # Train models
        results = self.detector.train(X, y, 'kdd99')
        
        # Simulate real-time detection on test data
        test_sample = X.sample(n=100, random_state=42)
        threats = self.detector.detect_threats_realtime(test_sample)
        
        logger.info(f"Detected {len(threats)} threats in real-time simulation")
        
        # Respond to threats
        for threat in threats[:5]:  # Limit to first 5 for demo
            response = self.responder.respond_to_dataset_threat(threat, 'kdd99')
            self.incident_log.append({'threat': threat, 'response': response})
        
        return {'results': results, 'threats': threats}
    
    def run_cicids_analysis(self):
        """Run complete analysis on CICIDS2017-like dataset"""
        logger.info("=== CICIDS2017 ANALYSIS ===")
        
        # Create sample dataset (replace with real CICIDS2017 when available)
        df = self.dataset_manager.create_sample_cicids2017()
        
        # Preprocess
        X, y = self.detector.preprocess_cicids(df)
        
        # Train models
        results = self.detector.train(X, y, 'cicids')
        
        # Simulate real-time detection
        test_sample = X.sample(n=100, random_state=42)
        threats = self.detector.detect_threats_realtime(test_sample)
        
        logger.info(f"Detected {len(threats)} threats in CICIDS simulation")
        
        # Respond to threats
        for threat in threats[:5]:  # Limit to first 5 for demo
            response = self.responder.respond_to_dataset_threat(threat, 'cicids')
            self.incident_log.append({'threat': threat, 'response': response})
        
        return {'results': results, 'threats': threats}
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        if not self.incident_log:
            return {'message': 'No incidents processed yet'}
        
        stats = {
            'total_incidents': len(self.incident_log),
            'dataset_distribution': {},
            'risk_distribution': {},
            'detection_methods': {},
            'attack_types': {}
        }
        
        for incident in self.incident_log:
            # Dataset distribution
            dataset = incident['response']['dataset_type']
            stats['dataset_distribution'][dataset] = stats['dataset_distribution'].get(dataset, 0) + 1
            
            # Risk distribution
            risk = incident['threat']['risk_level']
            stats['risk_distribution'][risk] = stats['risk_distribution'].get(risk, 0) + 1
            
            # Detection methods
            for method in incident['threat']['detected_by']:
                stats['detection_methods'][method] = stats['detection_methods'].get(method, 0) + 1
            
            # Attack types
            attack_type = incident['response']['attack_type']
            stats['attack_types'][attack_type] = stats['attack_types'].get(attack_type, 0) + 1
        
        return stats

# Demo execution
if __name__ == "__main__":
    logger.info("Starting Real Dataset Threat Detection System")
    
    # Initialize system
    system = RealDatasetThreatSystem()
    
    try:
        # Run KDD99 analysis
        kdd_results = system.run_kdd99_analysis()
        
        # Run CICIDS analysis
        cicids_results = system.run_cicids_analysis()
        
        # Display comprehensive statistics
        stats = system.get_comprehensive_stats()
        
        print("\n" + "="*60)
        print("COMPREHENSIVE THREAT DETECTION RESULTS")
        print("="*60)
        
        print(f"Total incidents processed: {stats['total_incidents']}")
        print(f"Dataset distribution: {stats['dataset_distribution']}")
        print(f"Risk level distribution: {stats['risk_distribution']}")
        print(f"Detection methods used: {stats['detection_methods']}")
        print(f"Attack types identified: {stats['attack_types']}")
        
        print("\n=== RECENT INCIDENTS ===")
        for i, incident in enumerate(system.incident_log[-3:], 1):
            print(f"\nIncident #{i}:")
            print(f"  Threat ID: {incident['response']['threat_id']}")
            print(f"  Dataset: {incident['response']['dataset_type']}")
            print(f"  Risk Level: {incident['threat']['risk_level']}")
            print(f"  Attack Type: {incident['response']['attack_type']}")
            print(f"  Detected By: {', '.join(incident['threat']['detected_by'])}")
            print(f"  Actions: {len(incident['response']['actions_taken'])} taken")
        
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"Error occurred: {e}")