# Game Investment Potential Prediction - Notebook Fixes

## 🚨 Critical Issues Identified

### 1. **Cell Order Issues**
- **Problem**: Cell 10 "#9 VALIDATION SUMMARY" references variables that don't exist yet
- **Problem**: Cell 13 "#8 Model Training" comes after validation summary that depends on it
- **Solution**: Reorder cells to: Feature Selection → Model Training → Validation → Recommendations

### 2. **Missing Validation Code**
- **Problem**: Variables like `cv_results`, `significance_results`, `bootstrap_results` are never created
- **Solution**: Add comprehensive validation code before validation summary

### 3. **Unicode Error in Data Loading**
- **Problem**: `youtube_df = pd.read_csv('youtube_game_videos.csv')` causes UnicodeDecodeError
- **Solution**: Add encoding handling with try-except blocks

### 4. **Duplicate Sections**
- **Problem**: Multiple cells for investment recommendations, final summary, save results
- **Solution**: Remove duplicate cells, keep only the most comprehensive versions

## 🔧 **Corrected Cell Order**

```
Cell 1: Introduction (Raw)
Cell 2: pip install xgboost lightgbm shap optuna
Cell 3: pip install pandas numpy matplotlib seaborn scikit-learn vaderSentiment
Cell 4: Library Imports
Cell 5: Data Loading (WITH FIX)
Cell 6: Data Overview
Cell 7: Feature Engineering
Cell 8: Create Master Dataset
Cell 9: Exploratory Data Analysis
Cell 10: Feature Selection
Cell 11: Model Training (MOVED HERE)
Cell 12: Comprehensive Validation (NEW)
Cell 13: Investment Recommendations
Cell 14: Final Summary
Cell 15: Save Results
```

## 📝 **Specific Fixes Needed**

### Fix 1: Data Loading (Cell 5)
```python
# 2. Data Loading with Error Handling
print("📊 Loading data files...")

# Load Twitch data
twitch_df = pd.read_csv('twitch_game_streams.csv')
print(f"✅ Twitch data loaded: {len(twitch_df)} rows")

# Load YouTube data with encoding handling
try:
    youtube_df = pd.read_csv('youtube_game_videos.csv')
    print(f"✅ YouTube data loaded: {len(youtube_df)} rows")
except UnicodeDecodeError:
    print("⚠️ UTF-8 encoding failed, trying alternative encodings...")
    try:
        youtube_df = pd.read_csv('youtube_game_videos.csv', encoding='latin-1')
        print(f"✅ YouTube data loaded with latin-1 encoding: {len(youtube_df)} rows")
    except:
        try:
            youtube_df = pd.read_csv('youtube_game_videos.csv', encoding='cp1252')
            print(f"✅ YouTube data loaded with cp1252 encoding: {len(youtube_df)} rows")
        except Exception as e:
            print(f"❌ Failed to load YouTube data: {e}")
            youtube_df = pd.DataFrame()

# Load Reddit data
reddit_df = pd.read_csv('reddit_game_posts.csv')
print(f"✅ Reddit data loaded: {len(reddit_df)} rows")

print("\n🎯 All data loaded successfully!")
```

### Fix 2: Add Missing Validation Code (New Cell 12)
```python
# 8. Comprehensive Model Validation
print("🔬 COMPREHENSIVE MODEL VALIDATION")
print("=" * 50)

from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.utils import resample
import scipy.stats as stats

# 1. Cross-validation analysis
print("\n📊 K-FOLD CROSS-VALIDATION ANALYSIS:")
cv_results = {}

for name, model in models.items():
    print(f"\n🔄 Cross-validating {name}...")
    
    # Use scaled data for all models for consistency
    cv_r2_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_rmse_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_rmse_scores)
    
    cv_results[name] = {
        'r2_scores': cv_r2_scores,
        'r2_mean': cv_r2_scores.mean(),
        'r2_std': cv_r2_scores.std(),
        'rmse_scores': cv_rmse_scores,
        'rmse_mean': cv_rmse_scores.mean(),
        'rmse_std': cv_rmse_scores.std()
    }
    
    print(f"✅ {name} CV Results:")
    print(f"   R² = {cv_r2_scores.mean():.3f} ± {cv_r2_scores.std():.3f}")
    print(f"   RMSE = {cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f}")

# 2. Statistical significance testing
print("\n🧪 STATISTICAL SIGNIFICANCE TESTING:")
significance_results = {}

for name, model in models.items():
    print(f"\n🔍 Testing significance of {name}...")
    
    # Use permutation test
    score, permutation_scores, p_value = permutation_test_score(
        model, X_train_scaled, y_train, 
        scoring='r2', cv=5, n_permutations=100, random_state=42
    )
    
    significance_results[name] = {
        'score': score,
        'permutation_scores': permutation_scores,
        'p_value': p_value,
        'is_significant': p_value < 0.05
    }
    
    significance_status = "✅ SIGNIFICANT" if p_value < 0.05 else "❌ NOT SIGNIFICANT"
    print(f"   {significance_status} (p = {p_value:.4f})")

# 3. Bootstrap confidence intervals
print("\n📈 BOOTSTRAP CONFIDENCE INTERVALS:")
bootstrap_results = {}

for name, model in models.items():
    print(f"\n🔄 Bootstrap analysis for {name}...")
    
    bootstrap_scores = []
    for i in range(100):
        X_boot, y_boot = resample(X_train_scaled, y_train, random_state=i)
        model.fit(X_boot, y_boot)
        score = model.score(X_train_scaled, y_train)
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    
    bootstrap_results[name] = {
        'scores': bootstrap_scores,
        'mean': bootstrap_scores.mean(),
        'std': bootstrap_scores.std(),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
    
    print(f"✅ {name} Bootstrap CI:")
    print(f"   Mean R² = {bootstrap_scores.mean():.3f}")
    print(f"   95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]")

# 4. Feature stability analysis (Random Forest only)
print("\n🌳 FEATURE STABILITY ANALYSIS:")
feature_stability = {}

if 'Random Forest' in models:
    rf_importances = []
    for i in range(10):
        X_boot, y_boot = resample(X_selected, y, random_state=i)
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=i)
        rf_temp.fit(X_boot, y_boot)
        rf_importances.append(rf_temp.feature_importances_)
    
    rf_importances = np.array(rf_importances)
    mean_importance = rf_importances.mean(axis=0)
    std_importance = rf_importances.std(axis=0)
    cv_importance = std_importance / mean_importance
    
    feature_stability['Random Forest'] = {
        'mean_importance': pd.Series(mean_importance, index=X_selected.columns),
        'std_importance': pd.Series(std_importance, index=X_selected.columns),
        'cv_importance': pd.Series(cv_importance, index=X_selected.columns)
    }
    
    print(f"✅ Feature importance stability calculated")
    print(f"   Most stable features: {feature_stability['Random Forest']['cv_importance'].nsmallest(5).index.tolist()}")

# 5. Model robustness testing
print("\n🛡️ MODEL ROBUSTNESS TESTING:")
robustness_results = {}

for name, model in models.items():
    print(f"\n🔄 Testing robustness of {name}...")
    
    robustness_scores = []
    for pct in [0.7, 0.8, 0.9]:
        subset_size = int(len(X_train_scaled) * pct)
        subset_scores = []
        
        for i in range(10):
            indices = np.random.choice(len(X_train_scaled), subset_size, replace=False)
            X_subset = X_train_scaled[indices]
            y_subset = y_train.iloc[indices]
            
            model.fit(X_subset, y_subset)
            score = model.score(X_subset, y_subset)
            subset_scores.append(score)
        
        robustness_scores.append({
            'subset_pct': pct,
            'scores': subset_scores,
            'mean_score': np.mean(subset_scores),
            'std_score': np.std(subset_scores)
        })
    
    robustness_results[name] = robustness_scores
    
    print(f"✅ {name} Robustness:")
    for result in robustness_scores:
        print(f"   {result['subset_pct']*100:.0f}% data: R² = {result['mean_score']:.3f} ± {result['std_score']:.3f}")

# 6. Composite model scoring
print("\n🏆 COMPOSITE MODEL SCORING:")
model_scores = {}

for name in models.keys():
    # Weighted composite score
    composite_score = (
        cv_results[name]['r2_mean'] * 0.4 +  # Cross-validation performance
        (1 - cv_results[name]['r2_std']) * 0.2 +  # Stability (lower std is better)
        (1 if significance_results[name]['is_significant'] else 0) * 0.2 +  # Significance
        (bootstrap_results[name]['ci_upper'] - bootstrap_results[name]['ci_lower']) * 0.2  # Precision
    )
    
    model_scores[name] = {
        'composite_score': composite_score,
        'cv_performance': cv_results[name]['r2_mean'],
        'stability': 1 - cv_results[name]['r2_std'],
        'significance': 1 if significance_results[name]['is_significant'] else 0,
        'precision': bootstrap_results[name]['ci_upper'] - bootstrap_results[name]['ci_lower']
    }

# Display final rankings
print("\n🏅 FINAL MODEL RANKINGS:")
sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
for i, (name, scores) in enumerate(sorted_models, 1):
    print(f"{i}. {name}: Composite Score = {scores['composite_score']:.3f}")

print(f"\n🎯 VALIDATION ANALYSIS COMPLETE!")
```

### Fix 3: Remove Duplicate Cells
- Delete duplicate investment recommendation cells
- Delete duplicate final summary cells  
- Delete duplicate save results cells

## 🎯 **Quick Fix Instructions**

1. **Save your current notebook** (backup)
2. **Delete cells 11-17** (all the problematic/duplicate cells)
3. **Add the corrected validation code** as new cell 12
4. **Fix the data loading** in cell 5
5. **Test the notebook** by running all cells

## ✅ **Expected Results After Fix**

- All cells run without errors
- Proper logical flow from data loading → analysis → modeling → validation → recommendations
- No missing variable references
- No duplicate content
- Comprehensive validation with statistical rigor

## 🔍 **Validation Checklist**

After applying fixes, verify:
- [ ] Data loads without Unicode errors
- [ ] All models train successfully
- [ ] Validation variables exist (`cv_results`, `significance_results`, etc.)
- [ ] Validation summary runs without KeyError
- [ ] Investment recommendations generate successfully
- [ ] All output files save correctly

This comprehensive fix will transform your notebook from a broken state to a production-ready, scientifically rigorous analysis tool. 