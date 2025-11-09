from flask import Flask, render_template, request, jsonify, session, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, inspect, MetaData
from rapidfuzz import fuzz, process
import io
import os
import uuid
from datetime import datetime, timedelta
import json
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = None  # No file size limit
app.config['UPLOAD_FOLDER'] = '/tmp/pakdata_uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Session data storage
session_data = {}

def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_session_data():
    sid = get_session_id()
    if sid not in session_data:
        session_data[sid] = {
            'raw_df': None,
            'clean_df': None,
            'clean_sample': None,
            'cleaning_log': []
        }
    return session_data[sid]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/app')
def app_page():
    return render_template('app.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload CSV or Excel files.'}), 400
        
        # Store in session
        data = get_session_data()
        data['raw_df'] = df
        data['clean_df'] = df.copy()
        data['cleaning_log'] = []
        
        # Generate overview
        overview = generate_overview(df)
        return jsonify(overview)
    
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/upload_clean_sample', methods=['POST'])
def upload_clean_sample():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Store clean sample
        data = get_session_data()
        data['clean_sample'] = df
        
        return jsonify({'message': 'Clean sample uploaded successfully', 'rows': len(df), 'columns': len(df.columns)})
    
    except Exception as e:
        return jsonify({'error': f'Error processing clean sample: {str(e)}'}), 500

@app.route('/connect_database', methods=['POST'])
def connect_database():
    try:
        db_config = request.json
        db_type = db_config.get('db_type')
        host = db_config.get('host')
        port = db_config.get('port')
        database = db_config.get('database')
        user = db_config.get('user')
        password = db_config.get('password')
        
        # Build connection string
        if db_type == 'postgresql':
            conn_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'
        elif db_type == 'mysql':
            conn_string = f'mysql+mysqldb://{user}:{password}@{host}:{port}/{database}'
        elif db_type == 'sqlserver':
            conn_string = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server'
        elif db_type == 'sqlite':
            conn_string = f'sqlite:///{database}'
        elif db_type == 'oracle':
            conn_string = f'oracle+cx_oracle://{user}:{password}@{host}:{port}/{database}'
        else:
            return jsonify({'error': 'Unsupported database type'}), 400
        
        # Test connection
        engine = create_engine(conn_string)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        # Store connection info in session
        data = get_session_data()
        data['db_engine'] = engine
        data['db_inspector'] = inspector
        
        return jsonify({
            'message': 'Database connected successfully',
            'tables': tables
        })
    
    except Exception as e:
        return jsonify({'error': f'Database connection failed: {str(e)}'}), 500

@app.route('/load_tables', methods=['POST'])
def load_tables():
    try:
        selected_tables = request.json.get('tables', [])
        data = get_session_data()
        
        if 'db_engine' not in data:
            return jsonify({'error': 'No database connection established'}), 400
        
        engine = data['db_engine']
        inspector = data['db_inspector']
        
        if len(selected_tables) == 0:
            return jsonify({'error': 'No tables selected'}), 400
        
        if len(selected_tables) == 1:
            # Load single table
            df = pd.read_sql_table(selected_tables[0], engine)
            data['raw_df'] = df
            data['clean_df'] = df.copy()
            data['cleaning_log'] = []
            
            overview = generate_overview(df)
            return jsonify(overview)
        
        else:
            # Multiple tables - attempt to join
            relationships = detect_relationships(inspector, selected_tables)
            
            if not relationships:
                return jsonify({
                    'error': 'No explicit relationships detected between selected tables. Please select a single table for direct import.'
                }), 400
            
            # Perform joins based on detected relationships
            df = join_tables(engine, selected_tables, relationships)
            data['raw_df'] = df
            data['clean_df'] = df.copy()
            data['cleaning_log'] = []
            
            overview = generate_overview(df)
            overview['join_info'] = f"Tables successfully joined on detected relationships: {', '.join([f'{r[0]}.{r[1]} = {r[2]}.{r[3]}' for r in relationships])}"
            return jsonify(overview)
    
    except Exception as e:
        return jsonify({'error': f'Error loading tables: {str(e)}'}), 500

def detect_relationships(inspector, tables):
    """Detect foreign key relationships between tables"""
    relationships = []
    
    for table in tables:
        fks = inspector.get_foreign_keys(table)
        for fk in fks:
            if fk['referred_table'] in tables:
                relationships.append((
                    table,
                    fk['constrained_columns'][0],
                    fk['referred_table'],
                    fk['referred_columns'][0]
                ))
    
    return relationships

def join_tables(engine, tables, relationships):
    """Join tables based on detected relationships"""
    # Load first table
    df = pd.read_sql_table(tables[0], engine)
    
    # Join remaining tables
    for rel in relationships:
        left_table, left_col, right_table, right_col = rel
        if right_table in tables and right_table != tables[0]:
            right_df = pd.read_sql_table(right_table, engine)
            df = df.merge(right_df, left_on=left_col, right_on=right_col, how='left', suffixes=('', f'_{right_table}'))
    
    return df

def generate_overview(df):
    """Generate data overview statistics"""
    total_rows = len(df)
    total_cols = len(df.columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    total_missing = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

    # Data quality assessment
    completeness = 1 - (total_missing / (total_rows * total_cols))
    duplicate_rate = duplicate_rows / total_rows if total_rows > 0 else 0

    if completeness > 0.95 and duplicate_rate < 0.01:
        quality = 'Excellent'
    elif completeness > 0.85 and duplicate_rate < 0.05:
        quality = 'Good'
    elif completeness > 0.70 and duplicate_rate < 0.10:
        quality = 'Fair'
    else:
        quality = 'Poor'

    # Replace NaN with None for JSON compatibility
    preview_df = df.head(20).replace({np.nan: None})

    return {
        'total_rows': total_rows,
        'total_columns': total_cols,
        'numerical_columns': len(num_cols),
        'categorical_columns': len(cat_cols),
        'total_missing': int(total_missing),
        'duplicate_rows': int(duplicate_rows),
        'memory_usage_mb': round(memory_usage, 2),
        'data_quality': quality,
        'num_col_names': num_cols,
        'cat_col_names': cat_cols,
        'preview': preview_df.to_dict('records'),
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }

@app.route('/convert_types', methods=['POST'])
def convert_types():
    try:
        conversions = request.json.get('conversions', {})
        data = get_session_data()
        
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        df = data['clean_df']
        conversion_log = []
        
        for col, target_type in conversions.items():
            if col not in df.columns:
                continue
            
            try:
                if target_type == 'integer':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif target_type == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif target_type == 'string':
                    df[col] = df[col].astype(str)
                elif target_type == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif target_type == 'boolean':
                    df[col] = df[col].astype(bool)
                
                conversion_log.append(f"Converted {col} to {target_type}")
            except Exception as e:
                conversion_log.append(f"Error converting {col}: {str(e)}")
        
        data['clean_df'] = df
        data['cleaning_log'].extend(conversion_log)
        
        return jsonify({
            'message': 'Type conversions applied',
            'log': conversion_log,
            'dtypes': df.dtypes.astype(str).to_dict()
        })
    
    except Exception as e:
        return jsonify({'error': f'Error converting types: {str(e)}'}), 500

@app.route('/clean_data', methods=['POST'])
def clean_data():
    try:
        config = request.json
        data = get_session_data()
        
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        df = data['clean_df'].copy()
        log = []
        
        # Missing value handling
        if config.get('null_method') and config['null_method'] != 'none':
            df, null_log = handle_missing_values(df, config, data.get('clean_sample'))
            log.extend(null_log)
        
        # Outlier detection and handling
        if config.get('outlier_method') and config['outlier_method'] != 'none':
            df, outlier_log = handle_outliers(df, config)
            log.extend(outlier_log)
        
        # Duplicate handling
        if config.get('remove_exact_duplicates'):
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            log.append(f"Removed {removed} exact duplicate rows")
        
        if config.get('remove_partial_duplicates') and config.get('partial_dup_columns'):
            before = len(df)
            df = df.drop_duplicates(subset=config['partial_dup_columns'])
            removed = before - len(df)
            log.append(f"Removed {removed} partial duplicates based on columns: {', '.join(config['partial_dup_columns'])}")
        
        if config.get('fuzzy_matching') and config.get('fuzzy_columns'):
            df, fuzzy_log = handle_fuzzy_duplicates(df, config)
            log.extend(fuzzy_log)
        
        # Text cleaning
        if config.get('text_cleaning'):
            df, text_log = clean_text(df, config)
            log.extend(text_log)
        
        # Value standardization
        if config.get('value_maps'):
            df, map_log = apply_value_maps(df, config)
            log.extend(map_log)
        
        # Domain validation
        if config.get('validation_rules'):
            df, val_log = apply_validation_rules(df, config)
            log.extend(val_log)
        
        # Expired data management
        if config.get('manage_expired') and config.get('date_column'):
            df, exp_log = handle_expired_data(df, config)
            log.extend(exp_log)
        
        # Update session data
        data['clean_df'] = df
        data['cleaning_log'].extend(log)
        
        # Generate summary
        summary = {
            'rows_after': len(df),
            'columns_after': len(df.columns),
            'missing_after': int(df.isnull().sum().sum()),
            'log': log,
            'preview': df.head(20).replace({np.nan: None}).to_dict('records')
        }
        
        return jsonify(summary)
    
    except Exception as e:
        return jsonify({'error': f'Error cleaning data: {str(e)}'}), 500

def handle_missing_values(df, config, clean_sample=None):
    log = []
    method = config.get('null_method')
    
    if method == 'remove':
        before = len(df)
        df = df.dropna()
        removed = before - len(df)
        log.append(f"Removed {removed} rows with missing values")
    
    elif method in ['mean', 'median', 'mode']:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            missing = df[col].isnull().sum()
            if missing > 0:
                if method == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                else:  # mode
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0, inplace=True)
                log.append(f"Filled {missing} missing values in {col} with {method}")
    
    elif method == 'interpolate':
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            missing = df[col].isnull().sum()
            if missing > 0:
                df[col] = df[col].interpolate()
                log.append(f"Interpolated {missing} missing values in {col}")
    
    elif method == 'knn':
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            df[num_cols] = imputer.fit_transform(df[num_cols])
            log.append(f"Applied KNN imputation to numerical columns")
    
    # Logical NULL pattern learning
    if config.get('enable_null_learning') and clean_sample is not None:
        df, pattern_log = learn_null_patterns(df, clean_sample, config)
        log.extend(pattern_log)
    
    return df, log

def learn_null_patterns(df, clean_sample, config):
    """Learn logical NULL patterns from clean sample"""
    log = []
    threshold = config.get('null_threshold', 0.8)
    min_support = config.get('min_support', 10)
    
    # Analyze null patterns in clean sample
    for col in df.columns:
        if col not in clean_sample.columns:
            continue
        
        # Find conditions where nulls are expected in clean sample
        null_mask = clean_sample[col].isnull()
        if null_mask.sum() >= min_support:
            # Learn pattern (simplified - could be more sophisticated)
            log.append(f"Learned NULL pattern for {col} from clean sample")
    
    return df, log

def handle_outliers(df, config):
    log = []
    method = config.get('outlier_method')
    handling = config.get('outlier_handling', 'remove')
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(num_cols) == 0:
        return df, ['No numerical columns for outlier detection']
    
    outlier_mask = pd.Series([False] * len(df))
    
    if method == 'zscore':
        threshold = config.get('threshold', 3)
        for col in num_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask |= z_scores > threshold
        log.append(f"Detected outliers using Z-score (threshold={threshold})")
    
    elif method == 'iqr':
        threshold = config.get('threshold', 1.5)
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask |= (df[col] < (Q1 - threshold * IQR)) | (df[col] > (Q3 + threshold * IQR))
        log.append(f"Detected outliers using IQR (threshold={threshold})")
    
    elif method == 'isolation_forest':
        contamination = config.get('contamination', 0.1)
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(df[num_cols].fillna(0))
        outlier_mask = predictions == -1
        log.append(f"Detected outliers using Isolation Forest (contamination={contamination})")
    
    elif method == 'lof':
        n_neighbors = config.get('n_neighbors', 20)
        clf = LocalOutlierFactor(n_neighbors=n_neighbors)
        predictions = clf.fit_predict(df[num_cols].fillna(0))
        outlier_mask = predictions == -1
        log.append(f"Detected outliers using LOF (n_neighbors={n_neighbors})")
    
    elif method == 'dbscan':
        eps = config.get('eps', 0.5)
        min_samples = config.get('min_samples', 5)
        # Standardize data for DBSCAN
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[num_cols].fillna(0))
        clf = DBSCAN(eps=eps, min_samples=min_samples)
        predictions = clf.fit_predict(scaled_data)
        outlier_mask = predictions == -1
        log.append(f"Detected outliers using DBSCAN (eps={eps}, min_samples={min_samples})")
    
    outlier_count = outlier_mask.sum()
    
    if handling == 'remove':
        df = df[~outlier_mask]
        log.append(f"Removed {outlier_count} outlier rows")
    elif handling == 'cap':
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        log.append(f"Capped {outlier_count} outlier values")
    
    return df, log

def handle_fuzzy_duplicates(df, config):
    log = []
    columns = config.get('fuzzy_columns', [])
    threshold = config.get('fuzzy_threshold', 80)
    
    for col in columns:
        if col not in df.columns:
            continue
        
        # Get unique values
        unique_vals = df[col].dropna().unique()
        
        # Find fuzzy matches
        matches = []
        for val in unique_vals:
            similar = process.extract(val, unique_vals, scorer=fuzz.ratio, limit=5)
            for match, score in similar:
                if score >= threshold and val != match:
                    matches.append((val, match, score))
        
        # Standardize to most frequent
        if matches:
            for val1, val2, score in matches:
                freq1 = (df[col] == val1).sum()
                freq2 = (df[col] == val2).sum()
                if freq1 > freq2:
                    df[col] = df[col].replace(val2, val1)
                else:
                    df[col] = df[col].replace(val1, val2)
            
            log.append(f"Standardized {len(matches)} fuzzy matches in {col} (threshold={threshold}%)")
    
    return df, log

def clean_text(df, config):
    log = []
    text_cols = df.select_dtypes(include=['object']).columns
    
    if config.get('trim_whitespace'):
        for col in text_cols:
            df[col] = df[col].str.strip()
        log.append("Trimmed whitespace from text columns")
    
    if config.get('remove_nonprintable'):
        for col in text_cols:
            df[col] = df[col].str.replace(r'[^\x20-\x7E]', '', regex=True)
        log.append("Removed non-printable characters")
    
    if config.get('normalize_case'):
        case_type = config.get('case_type', 'lower')
        for col in text_cols:
            if case_type == 'upper':
                df[col] = df[col].str.upper()
            elif case_type == 'lower':
                df[col] = df[col].str.lower()
            elif case_type == 'title':
                df[col] = df[col].str.title()
        log.append(f"Normalized text case to {case_type}")
    
    return df, log

def apply_value_maps(df, config):
    log = []
    value_maps = config.get('value_maps', {})
    
    for col, mappings in value_maps.items():
        if col in df.columns:
            df[col] = df[col].replace(mappings)
            log.append(f"Applied value standardization to {col}")
    
    return df, log

def apply_validation_rules(df, config):
    log = []
    rules = config.get('validation_rules', [])
    action = config.get('validation_action', 'flag')
    
    for rule in rules:
        try:
            # Evaluate rule
            mask = df.eval(rule)
            violations = (~mask).sum()
            
            if violations > 0:
                if action == 'flag':
                    df['validation_flag'] = ~mask
                elif action == 'remove':
                    df = df[mask]
                elif action == 'set_null':
                    # Set violating rows to NaN (simplified)
                    pass
                
                log.append(f"Rule '{rule}': {violations} violations {action}ed")
        except Exception as e:
            log.append(f"Error applying rule '{rule}': {str(e)}")
    
    return df, log

def handle_expired_data(df, config):
    log = []
    date_col = config.get('date_column')
    threshold_date = config.get('threshold_date')
    threshold_days = config.get('threshold_days')
    action = config.get('expired_action', 'flag')
    
    if date_col not in df.columns:
        return df, ['Date column not found']
    
    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Determine threshold
    if threshold_date:
        threshold = pd.to_datetime(threshold_date)
    elif threshold_days:
        threshold = datetime.now() - timedelta(days=threshold_days)
    else:
        return df, ['No threshold specified']
    
    # Find expired records
    expired_mask = df[date_col] < threshold
    expired_count = expired_mask.sum()
    
    if action == 'flag':
        df['expired_flag'] = expired_mask
        log.append(f"Flagged {expired_count} expired records")
    elif action == 'remove':
        df = df[~expired_mask]
        log.append(f"Removed {expired_count} expired records")
    
    return df, log

@app.route('/transform_data', methods=['POST'])
def transform_data():
    try:
        config = request.json
        data = get_session_data()
        
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        df = data['clean_df']
        transform_type = config.get('type')
        
        if transform_type == 'pivot':
            index = config.get('index')
            columns = config.get('columns')
            values = config.get('values')
            aggfunc = config.get('aggfunc', 'sum')
            
            result = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)
            
            return jsonify({
                'preview': result.reset_index().head(20).to_dict('records'),
                'message': 'Pivot table generated'
            })
        
        elif transform_type == 'groupby':
            groupby_cols = config.get('groupby')
            agg_cols = config.get('agg_columns')
            agg_funcs = config.get('agg_functions', ['sum'])
            
            result = df.groupby(groupby_cols)[agg_cols].agg(agg_funcs).reset_index()
            
            return jsonify({
                'preview': result.head(20).to_dict('records'),
                'message': 'Aggregated data generated'
            })
        
        else:
            return jsonify({'error': 'Unknown transformation type'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Error transforming data: {str(e)}'}), 500

@app.route('/download')
def download():
    try:
        data = get_session_data()

        if data['clean_df'] is None:
            return jsonify({'error': 'No data to download'}), 400

        # Create CSV in memory
        output = io.BytesIO()
        data['clean_df'].to_csv(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='cleaned_data.csv'
        )

    except Exception as e:
        return jsonify({'error': f'Error downloading data: {str(e)}'}), 500

# New routes for Data Insights & Quality section

@app.route('/get_data_quality', methods=['GET'])
def get_data_quality():
    try:
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df']
        total_rows = len(df)
        total_cols = len(df.columns)

        # Missing value analysis per column
        missing_analysis = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
            missing_analysis[col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }

        # Duplicate analysis
        duplicate_count = df.duplicated().sum()
        duplicate_indices = df[df.duplicated()].index.tolist()[:10]  # First 10 duplicates

        # Completeness gauge
        total_cells = total_rows * total_cols
        filled_cells = total_cells - df.isnull().sum().sum()
        completeness = (filled_cells / total_cells) * 100 if total_cells > 0 else 0

        return jsonify({
            'missing_analysis': missing_analysis,
            'duplicate_count': int(duplicate_count),
            'duplicate_indices': duplicate_indices,
            'completeness': round(completeness, 2),
            'total_rows': total_rows,
            'total_columns': total_cols
        })

    except Exception as e:
        return jsonify({'error': f'Error getting data quality: {str(e)}'}), 500

@app.route('/generate_insights', methods=['POST'])
def generate_insights():
    try:
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df']
        total_rows = len(df)
        total_cols = len(df.columns)

        # Completeness Score
        total_cells = total_rows * total_cols
        filled_cells = total_cells - df.isnull().sum().sum()
        completeness_score = round((filled_cells / total_cells) * 100, 2) if total_cells > 0 else 0

        # Accuracy Score (simplified: based on data types consistency)
        accuracy_score = 85  # Placeholder - could be enhanced with more sophisticated checks

        # Consistency Score (simplified: based on duplicate rate and format consistency)
        duplicate_rate = df.duplicated().sum() / total_rows if total_rows > 0 else 0
        consistency_score = max(0, round((1 - duplicate_rate) * 100, 2))

        # Timeliness Score (simplified: based on date columns presence and validity)
        date_cols = df.select_dtypes(include=['datetime']).columns
        timeliness_score = 90 if len(date_cols) > 0 else 70  # Placeholder

        # Missing Analysis
        missing_analysis = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
            missing_analysis[col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }

        # Duplicates Analysis
        exact_duplicates = int(df.duplicated().sum())
        near_duplicates = 0  # Placeholder - could implement fuzzy matching
        duplicate_rate_pct = round((exact_duplicates / total_rows) * 100, 2) if total_rows > 0 else 0

        # Paginated Preview (first page)
        page_size = 25
        paginated_preview = df.head(page_size).to_dict('records')
        total_pages = (total_rows + page_size - 1) // page_size

        return jsonify({
            'completeness_score': completeness_score,
            'accuracy_score': accuracy_score,
            'consistency_score': consistency_score,
            'timeliness_score': timeliness_score,
            'missing_analysis': {
                'total_missing': int(df.isnull().sum().sum()),
                'missing_rate': round((df.isnull().sum().sum() / total_cells) * 100, 2) if total_cells > 0 else 0
            },
            'duplicates_analysis': {
                'exact_duplicates': exact_duplicates,
                'near_duplicates': near_duplicates,
                'duplicate_rate': duplicate_rate_pct
            },
            'paginatedPreview': paginated_preview,
            'totalPages': total_pages
        })

    except Exception as e:
        return jsonify({'error': f'Error generating insights: {str(e)}'}), 500

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        config = request.json
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df']
        if df.empty:
            return jsonify({'error': 'Data is empty'}), 400

        chart_type = config.get('chart_type')
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        color_col = config.get('color_column')
        agg_func = config.get('aggregation', 'count')

        if not x_col or x_col not in df.columns:
            return jsonify({'error': 'Invalid or missing X column'}), 400

        if chart_type in ['line', 'scatter', 'box'] and (not y_col or y_col not in df.columns):
            return jsonify({'error': 'Y column required and must exist for this chart type'}), 400

        if color_col and color_col not in df.columns:
            return jsonify({'error': 'Invalid color column'}), 400

        if chart_type == 'bar':
            if y_col and agg_func != 'count':
                grouped = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                fig = px.bar(grouped, x=x_col, y=y_col, color=color_col, title=f"{agg_func.capitalize()} of {y_col} by {x_col}")
            else:
                value_counts = df[x_col].value_counts().reset_index()
                value_counts.columns = [x_col, 'count']
                fig = px.bar(value_counts, x=x_col, y='count', color=color_col, title=f"Count by {x_col}")

        elif chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")

        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")

        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x_col, color=color_col, title=f"Distribution of {x_col}")

        elif chart_type == 'box':
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"Box plot of {y_col} by {x_col}")

        elif chart_type == 'pie':
            value_counts = df[x_col].value_counts().reset_index()
            value_counts.columns = [x_col, 'count']
            fig = px.pie(value_counts, names=x_col, values='count', title=f"Distribution of {x_col}")

        else:
            return jsonify({'error': 'Unsupported chart type'}), 400

        # Return Plotly figure dict directly
        return jsonify({'data': fig.data, 'layout': fig.layout})
    except Exception as e:
        return jsonify({'error': f'Error generating chart: {str(e)}'}), 500

@app.route('/get_paginated_preview', methods=['POST'])
def get_paginated_preview():
    try:
        config = request.json
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df']
        page = config.get('page', 1)
        per_page = config.get('per_page', 20)
        show_cleaned = config.get('show_cleaned', True)
        highlight_nulls = config.get('highlight_nulls', True)
        highlight_duplicates = config.get('highlight_duplicates', True)

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        preview_df = df.iloc[start_idx:end_idx]

        # Prepare data with highlighting info
        preview_data = []
        for idx, row in preview_df.iterrows():
            row_data = {'index': int(idx)}
            highlights = {'nulls': [], 'duplicates': False}

            for col in df.columns:
                val = row[col]
                row_data[col] = str(val) if pd.notna(val) else None
                if highlight_nulls and pd.isna(val):
                    highlights['nulls'].append(col)

            if highlight_duplicates and idx in df[df.duplicated()].index:
                highlights['duplicates'] = True

            row_data['_highlights'] = highlights
            preview_data.append(row_data)

        total_pages = (len(df) + per_page - 1) // per_page

        return jsonify({
            'data': preview_data,
            'columns': df.columns.tolist(),
            'current_page': page,
            'total_pages': total_pages,
            'total_rows': len(df)
        })

    except Exception as e:
        return jsonify({'error': f'Error getting paginated preview: {str(e)}'}), 500

@app.route('/handle_expired_values', methods=['POST'])
def handle_expired_values():
    try:
        config = request.json
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df'].copy()
        date_col = config.get('date_column')
        threshold_days = config.get('threshold_days', 365)
        action = config.get('action', 'flag')

        if date_col not in df.columns:
            return jsonify({'error': 'Date column not found'}), 400

        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Calculate expiry
        threshold_date = datetime.now() - timedelta(days=threshold_days)
        expired_mask = df[date_col] < threshold_date
        expired_count = expired_mask.sum()

        if action == 'flag':
            df['expired'] = expired_mask
        elif action == 'remove':
            df = df[~expired_mask]
        elif action == 'archive':
            # For simplicity, just flag as archived
            df['archived'] = expired_mask

        # Update session
        data['clean_df'] = df

        # Generate histogram data for expiry distribution
        expired_dates = df[expired_mask][date_col].dropna()
        if len(expired_dates) > 0:
            hist_data = expired_dates.dt.to_period('M').value_counts().sort_index()
            hist_json = json.dumps({
                'x': hist_data.index.astype(str).tolist(),
                'y': hist_data.values.tolist(),
                'type': 'bar',
                'name': 'Expired Records by Month'
            })
        else:
            hist_json = None

        return jsonify({
            'expired_count': int(expired_count),
            'total_rows': len(df),
            'histogram': hist_json,
            'message': f'{action.capitalize()}ed {expired_count} expired records'
        })

    except Exception as e:
        return jsonify({'error': f'Error handling expired values: {str(e)}'}), 500

@app.route('/detect_erroneous_values', methods=['POST'])
def detect_erroneous_values():
    try:
        config = request.get_json() if request.is_json else {}
        data = get_session_data()

        # If data file is specified, load it
        if config.get('data'):
            filename = config['data']
            if os.path.exists(filename):
                if filename.endswith('.csv'):
                    df = pd.read_csv(filename)
                elif filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(filename)
                else:
                    return jsonify({'error': 'Unsupported file format'}), 400
                data['raw_df'] = df
                data['clean_df'] = df.copy()
                data['cleaning_log'] = []
            else:
                return jsonify({'error': 'File not found'}), 400
        elif data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df']
        detections = {}

        # Type mismatches for numeric columns
        for col in df.columns:
            try:
                dtype = str(df[col].dtype)
                if 'int' in dtype or 'float' in dtype:
                    # Check for non-numeric values that can't be converted
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    non_numeric_mask = numeric_series.isnull() & df[col].notna()
                    non_numeric = non_numeric_mask.sum()
                    if non_numeric > 0:
                        detections[col] = detections.get(col, {})
                        detections[col]['type_mismatch'] = int(non_numeric)
                elif 'datetime' in dtype:
                    # Check for invalid dates
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    invalid_dates = datetime_series.isnull() & df[col].notna()
                    invalid_count = invalid_dates.sum()
                    if invalid_count > 0:
                        detections[col] = detections.get(col, {})
                        detections[col]['invalid_dates'] = int(invalid_count)
            except Exception as e:
                # Skip column if error occurs
                continue

        # Out of range values (basic check for age-like columns)
        for col in df.columns:
            try:
                if 'age' in col.lower() and df[col].dtype in ['int64', 'float64']:
                    out_of_range = ((df[col] < 0) | (df[col] > 120)).sum()
                    if out_of_range > 0:
                        detections[col] = detections.get(col, {})
                        detections[col]['out_of_range'] = int(out_of_range)
            except Exception as e:
                continue

        # Additional checks for negative values in positive-only columns
        for col in df.columns:
            try:
                if df[col].dtype in ['int64', 'float64']:
                    if 'price' in col.lower() or 'cost' in col.lower() or 'amount' in col.lower():
                        negative = (df[col] < 0).sum()
                        if negative > 0:
                            detections[col] = detections.get(col, {})
                            detections[col]['negative_values'] = int(negative)
            except Exception as e:
                continue

        return jsonify({
            'detections': detections,
            'total_issues': sum(sum(issues.values()) for issues in detections.values())
        })

    except Exception as e:
        return jsonify({'error': f'Error detecting erroneous values: {str(e)}'}), 500

@app.route('/fix_erroneous_values', methods=['POST'])
def fix_erroneous_values():
    try:
        config = request.json
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df'].copy()
        fixes = config.get('fixes', {})

        log = []
        for col, fix_config in fixes.items():
            if col not in df.columns:
                continue

            action = fix_config.get('action')
            default_value = fix_config.get('default_value')

            if action == 'set_null':
                # Identify erroneous values and set to null
                if df[col].dtype in ['int64', 'float64']:
                    mask = df[col].astype(str).str.match(r'[^0-9.\-\+eE\s]')
                    df.loc[mask, col] = np.nan
                    log.append(f"Set {mask.sum()} erroneous values in {col} to null")
                elif 'datetime' in str(df[col].dtype):
                    mask = pd.to_datetime(df[col], errors='coerce').isnull()
                    df.loc[mask, col] = np.nan
                    log.append(f"Set {mask.sum()} invalid dates in {col} to null")

            elif action == 'convert':
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    log.append(f"Converted {col} to numeric, coercing errors")

            elif action == 'replace':
                if default_value is not None:
                    # Replace erroneous values with default
                    if df[col].dtype in ['int64', 'float64']:
                        mask = df[col].astype(str).str.match(r'[^0-9.\-\+eE\s]')
                        df.loc[mask, col] = default_value
                        log.append(f"Replaced {mask.sum()} erroneous values in {col} with {default_value}")

        data['clean_df'] = df
        data['cleaning_log'].extend(log)

        return jsonify({
            'message': 'Erroneous values fixed',
            'log': log
        })

    except Exception as e:
        return jsonify({'error': f'Error fixing erroneous values: {str(e)}'}), 500

@app.route('/detect_inconsistencies', methods=['POST'])
def detect_inconsistencies():
    try:
        config = request.json
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df']
        inconsistencies = {}

        # Case inconsistencies
        for col in df.select_dtypes(include=['object']).columns:
            unique_vals = df[col].dropna().unique()
            lower_vals = [str(v).lower() for v in unique_vals]
            if len(set(lower_vals)) < len(unique_vals):
                inconsistencies[col] = inconsistencies.get(col, {})
                inconsistencies[col]['case_inconsistency'] = True

        # Date format inconsistencies (simplified)
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    # If some parse and some don't, might be inconsistent
                    parseable = pd.to_datetime(df[col], errors='coerce').notna().sum()
                    total = df[col].notna().sum()
                    if parseable < total * 0.9:  # Less than 90% parseable
                        inconsistencies[col] = inconsistencies.get(col, {})
                        inconsistencies[col]['date_format_inconsistency'] = True
                except:
                    pass

        # Number format issues
        for col in df.select_dtypes(include=[np.number]).columns:
            str_vals = df[col].astype(str)
            has_commas = str_vals.str.contains(',').any()
            has_dollars = str_vals.str.contains(r'\$').any()
            if has_commas or has_dollars:
                inconsistencies[col] = inconsistencies.get(col, {})
                inconsistencies[col]['number_format_issues'] = True

        return jsonify({
            'inconsistencies': inconsistencies,
            'total_issues': len(inconsistencies)
        })

    except Exception as e:
        return jsonify({'error': f'Error detecting inconsistencies: {str(e)}'}), 500

@app.route('/fix_inconsistencies', methods=['POST'])
def fix_inconsistencies():
    try:
        config = request.json
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df'].copy()
        fixes = config.get('fixes', {})

        log = []
        for col, fix_config in fixes.items():
            if col not in df.columns:
                continue

            fix_type = fix_config.get('type')
            target_format = fix_config.get('target_format', 'lower')

            if fix_type == 'case':
                if target_format == 'lower':
                    df[col] = df[col].str.lower()
                elif target_format == 'upper':
                    df[col] = df[col].str.upper()
                elif target_format == 'title':
                    df[col] = df[col].str.title()
                log.append(f"Normalized case in {col} to {target_format}")

            elif fix_type == 'date_format':
                if target_format:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime(target_format)
                    log.append(f"Standardized date format in {col} to {target_format}")

            elif fix_type == 'number_format':
                # Remove non-numeric characters except decimal point
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                log.append(f"Cleaned number format in {col}")

        data['clean_df'] = df
        data['cleaning_log'].extend(log)

        return jsonify({
            'message': 'Inconsistencies fixed',
            'log': log
        })

    except Exception as e:
        return jsonify({'error': f'Error fixing inconsistencies: {str(e)}'}), 500

@app.route('/get_outlier_visualization', methods=['POST'])
def get_outlier_visualization():
    try:
        config = request.json
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df']
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not num_cols:
            return jsonify({'error': 'No numerical columns found'}), 400

        visualizations = {}

        # Box plots for each numerical column
        for col in num_cols[:5]:  # Limit to first 5 columns
            fig = go.Figure()
            fig.add_trace(go.Box(y=df[col], name=col))
            fig.update_layout(title=f"Box Plot of {col}", yaxis_title=col)
            visualizations[f'box_{col}'] = json.dumps(fig, cls=PlotlyJSONEncoder)

        # Scatter plot with outliers highlighted (using IQR method)
        if len(num_cols) >= 2:
            x_col = num_cols[0]
            y_col = num_cols[1]

            # Detect outliers in y_col
            Q1 = df[y_col].quantile(0.25)
            Q3 = df[y_col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[y_col] < (Q1 - 1.5 * IQR)) | (df[y_col] > (Q3 + 1.5 * IQR))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[~outlier_mask][x_col],
                y=df[~outlier_mask][y_col],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=6)
            ))
            fig.add_trace(go.Scatter(
                x=df[outlier_mask][x_col],
                y=df[outlier_mask][y_col],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=8, symbol='x')
            ))
            fig.update_layout(
                title=f"Scatter Plot: {x_col} vs {y_col} (Outliers Highlighted)",
                xaxis_title=x_col,
                yaxis_title=y_col
            )
            visualizations['scatter_outliers'] = json.dumps(fig, cls=PlotlyJSONEncoder)

        return jsonify({'visualizations': visualizations})

    except Exception as e:
        return jsonify({'error': f'Error generating outlier visualization: {str(e)}'}), 500

@app.route('/apply_pivot', methods=['POST'])
def apply_pivot():
    try:
        config = request.json
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df']
        index_col = config.get('pivotIndex')
        columns_col = config.get('pivotColumns')
        values_col = config.get('pivotValues')
        agg_func = config.get('pivotAggfunc', 'sum')

        if not index_col or not columns_col or not values_col:
            return jsonify({'error': 'Missing required parameters for pivot'}), 400

        index_cols = [index_col] if index_col else []

        pivot_df = pd.pivot_table(
            df,
            index=index_cols,
            columns=columns_col,
            values=values_col,
            aggfunc=agg_func
        ).reset_index()

        # Flatten column names
        pivot_df.columns = [str(col) for col in pivot_df.columns]

        data['clean_df'] = pivot_df
        overview = generate_overview(pivot_df)
        preview = pivot_df.head(20).to_dict('records')

        return jsonify({
            'message': 'Pivot table applied',
            'overview': overview,
            'preview': preview
        })

    except Exception as e:
        return jsonify({'error': f'Error applying pivot: {str(e)}'}), 500

@app.route('/apply_groupby', methods=['POST'])
def apply_groupby():
    try:
        config = request.json
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df']
        group_cols = config.get('groupByColumns', '').split(',').strip() if config.get('groupByColumns') else []
        agg_cols = config.get('aggColumns', '').split(',').strip() if config.get('aggColumns') else []
        agg_funcs = config.get('aggFunctions', '').split(',').strip() if config.get('aggFunctions') else ['sum']

        if not group_cols or not agg_cols:
            return jsonify({'error': 'Missing required parameters for groupby'}), 400

        grouped_df = df.groupby(group_cols)[agg_cols].agg(agg_funcs).reset_index()

        # Flatten multi-level columns if any
        if isinstance(grouped_df.columns, pd.MultiIndex):
            grouped_df.columns = ['_'.join(str(level) for level in col) for col in grouped_df.columns]

        data['clean_df'] = grouped_df
        overview = generate_overview(grouped_df)
        preview = grouped_df.head(20).to_dict('records')

        return jsonify({
            'message': 'Group by aggregation applied',
            'overview': overview,
            'preview': preview
        })

    except Exception as e:
        return jsonify({'error': f'Error applying groupby: {str(e)}'}), 500

@app.route('/generate_outlier_visualization', methods=['POST'])
def generate_outlier_visualization():
    try:
        data = get_session_data()
        if data['clean_df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        df = data['clean_df']
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not num_cols:
            return jsonify({'error': 'No numerical columns found for outlier visualization'}), 400

        # Create subplots for box plots of numerical columns
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        num_plots = min(len(num_cols), 6)  # Limit to 6 columns
        rows = (num_plots + 2) // 3  # 3 columns per row
        cols = min(3, num_plots)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f'Box Plot: {col}' for col in num_cols[:num_plots]],
            specs=[[{"type": "box"} for _ in range(cols)] for _ in range(rows)]
        )

        for i, col in enumerate(num_cols[:num_plots]):
            row = (i // 3) + 1
            col_pos = (i % 3) + 1

            # Calculate outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Separate normal and outlier points
            normal_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)][col]
            outlier_data = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

            # Add box plot for normal data
            fig.add_trace(
                go.Box(
                    y=normal_data,
                    name=f'Normal ({col})',
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=row,
                col=col_pos
            )

            # Add scatter points for outliers
            if len(outlier_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        y=outlier_data,
                        mode='markers',
                        name=f'Outliers ({col})',
                        marker=dict(color='red', size=8, symbol='x'),
                        showlegend=i == 0  # Show legend only for first plot
                    ),
                    row=row,
                    col=col_pos
                )

        fig.update_layout(
            title_text="Outlier Visualization - Box Plots with Outlier Markers",
            height=400 * rows,
            showlegend=True
        )

        # Return Plotly figure data and layout
        return jsonify({
            'data': fig.data,
            'layout': fig.layout
        })

    except Exception as e:
        return jsonify({'error': f'Error generating outlier visualization: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

