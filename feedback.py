import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

class FeedbackSystem:
    def __init__(self):
        self.thresholds = {}
        self.high_sq_patterns = []
        self.normal_sq_patterns = []
        self.feature_cols = []
        
    def fit(self, df, feature_cols, target_col='sleep_quality'):
        """
        Learns thresholds and mines frequent patterns.
        """
        self.feature_cols = feature_cols
        
        # 1. Learn Thresholds (Terciles)
        for col in feature_cols:
            if col == target_col:
                continue
            # Calculate terciles (33rd and 66th percentiles)
            low_th = df[col].quantile(0.33)
            high_th = df[col].quantile(0.66)
            self.thresholds[col] = (low_th, high_th)
            
        # 2. Discretize Data
        discretized_data = []
        for _, row in df.iterrows():
            discretized_row = self._discretize_row(row)
            # Add SQ level
            sq_val = row[target_col]
            # Define SQ groups based on paper or distribution
            # Paper: Low (251), Normal (998), High (322) -> roughly bottom 16%, top 20%?
            # Let's use simple terciles for SQ too for robustness
            sq_low_th = df[target_col].quantile(0.33)
            sq_high_th = df[target_col].quantile(0.66)
            
            if sq_val <= sq_low_th:
                sq_level = 'Low'
            elif sq_val <= sq_high_th:
                sq_level = 'Normal'
            else:
                sq_level = 'High'
                
            discretized_row['SQ_Level'] = sq_level
            discretized_data.append(discretized_row)
            
        df_disc = pd.DataFrame(discretized_data)
        
        # 3. Mine Patterns for High and Normal SQ
        self.high_sq_patterns = self._mine_patterns(df_disc[df_disc['SQ_Level'] == 'High'])
        self.normal_sq_patterns = self._mine_patterns(df_disc[df_disc['SQ_Level'] == 'Normal'])
        
        print(f"Mined {len(self.high_sq_patterns)} High SQ patterns.")
        print(f"Mined {len(self.normal_sq_patterns)} Normal SQ patterns.")

    def _discretize_row(self, row):
        """Converts a row of continuous features to categorical items."""
        items = {}
        for col in self.feature_cols:
            if col not in self.thresholds:
                continue
                
            val = row[col]
            low_th, high_th = self.thresholds[col]
            
            if val <= low_th:
                level = 'low'
            elif val <= high_th:
                level = 'normal'
            else:
                level = 'high'
            
            # Format: "col_level" e.g., "steps_low"
            items[col] = f"{col}_{level}"
        return items

    def _mine_patterns(self, df_group, min_support=0.2):
        """Mines frequent itemsets using Apriori."""
        if df_group.empty:
            return []
            
        # Convert to list of lists for TransactionEncoder
        transactions = []
        for _, row in df_group.iterrows():
            # Exclude SQ_Level from patterns
            items = [v for k, v in row.items() if k != 'SQ_Level']
            transactions.append(items)
            
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_trans = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
        
        # Sort by support and length
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        frequent_itemsets = frequent_itemsets.sort_values(['length', 'support'], ascending=[False, False])
        
        return frequent_itemsets['itemsets'].tolist()

    def generate_feedback(self, row, predicted_sq_val):
        """
        Generates feedback for a user based on their data and predicted SQ.
        """
        # Determine predicted SQ level
        # We need the SQ thresholds from fit. 
        # Ideally we should store them. Let's assume we can access them or re-estimate.
        # For simplicity, let's just assume if it's "Low" relative to general population.
        # But wait, we don't have the population distribution here.
        # Let's pass the thresholds or store them in __init__.
        # I'll update fit to store SQ thresholds.
        
        # Actually, let's just use the discretized row to see what the current state is.
        current_items = self._discretize_row(row)
        current_set = set(current_items.values())
        
        # Find best matching pattern in High/Normal groups
        # We want a pattern that is "close" to current state but "better".
        # "Better" usually means Normal or High levels for positive attributes (steps, sleep),
        # and Low or Normal for negative attributes (stress, fatigue).
        
        # Simple strategy: Find the longest pattern in High/Normal group that has high overlap.
        best_pattern = None
        max_overlap = -1
        
        # Search in High patterns first, then Normal
        target_patterns = self.high_sq_patterns + self.normal_sq_patterns
        
        for pattern in target_patterns:
            pattern_set = set(pattern)
            overlap = len(current_set.intersection(pattern_set))
            
            # We want a pattern that is NOT identical (otherwise no improvement)
            if overlap < len(pattern_set) and overlap > max_overlap:
                max_overlap = overlap
                best_pattern = pattern_set
        
        if not best_pattern:
            return "Keep up the good work!"

        # Identify differences
        feedback_msgs = []
        for item in best_pattern:
            if item not in current_set:
                # This is a target attribute we don't have.
                # item is like "steps_normal"
                feature, target_level = item.rsplit('_', 1)
                
                # Find current level
                current_level = "unknown"
                for curr_item in current_set:
                    if curr_item.startswith(feature + "_"):
                        _, current_level = curr_item.rsplit('_', 1)
                        break
                
                msg = self._get_feedback_message(feature, current_level, target_level)
                if msg:
                    feedback_msgs.append(msg)
                    
        if not feedback_msgs:
            return "Try to maintain a balanced lifestyle."
            
        return " | ".join(feedback_msgs)

    def _get_feedback_message(self, feature, current, target):
        """Returns a feedback message based on Table IV."""
        # Define feedback templates
        # Table IV mappings
        
        # Physical Activities
        if feature in ['steps', 'distance', 'calories', 'very_active_minutes', 'moderately_active_minutes']:
            if target in ['normal', 'high'] and current == 'low':
                if feature == 'steps':
                    return "Please try to walk more"
                elif feature == 'distance':
                    return "Let's go out and have a walk"
                elif feature == 'calories':
                    return "Let's do something to consume more calories"
                else:
                    return "Try to increase your physical activity intensity"
                    
        # Wellness
        if feature in ['fatigue', 'stress', 'soreness']:
            if target in ['normal', 'low'] and current == 'high':
                if feature == 'fatigue':
                    return "You seem tired, let's take some rest"
                elif feature == 'stress':
                    return "You may need some relax"
                elif feature == 'soreness':
                    return "Go easy on yourself, let's make time for hobbies"
                    
        if feature == 'mood':
            if target in ['normal', 'high'] and current == 'low':
                return "Let's adjust our mood, how about listening to happy music?"
                
        if feature == 'readiness':
             if target in ['normal', 'high'] and current == 'low':
                 return "How about taking a deep breath and making ourselves ready?"

        return None
