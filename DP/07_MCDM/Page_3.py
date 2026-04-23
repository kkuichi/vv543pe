import streamlit as st
from scipy.stats import rankdata
import pandas as pd
from sklearn.preprocessing import normalize, MinMaxScaler
import numpy as np
from pymcdm.methods import TOPSIS, ARAS, EDAS, MABAC, PROMETHEE_II, CODAS, MARCOS, WSM
from pymcdm.correlations import weighted_spearman
from pymcdm.helpers import correlation_matrix
from pyDecision.algorithm import ahp_method
from pymcdm.weights import critic_weights
from pymcdm import visuals
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

pd.set_option('display.max_rows', 500)
pastel_palette = [
    '#66c5cc', '#dcb0f2', '#f89c74', '#f6cf71', '#87c55f',
    '#e7d7ca', '#9eb9f3', '#fe88b1', '#c9db74', '#ddb398', '#e7d7ca'
]

st.title("Selecting the XAI method using MCDM")
st.divider()

df = None
uploaded_file = st.sidebar.file_uploader("UPLOAD YOUR DATA", type=["xlsx", "csv"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    try:
        if file_extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_extension == "csv":
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")

if df is not None:
    metrics_num = st.sidebar.number_input("ENTER A NUMBER OF METRICS", min_value=1, max_value=30, value=5)
    data = df.iloc[:, -metrics_num:]
    metric_name = data.columns.tolist()
    methods_name = df.iloc[:, 0]
    alternative_names = df.iloc[:, 0].tolist()
    num_alternatives = len(data)
    weights = []

    st.write("## Loaded data")
    st.write(df)

    st.sidebar.write("ENTER TYPES OF CRITERIA")
    pills_results = []
    if metrics_num > 0:
        for i in metric_name:
            selected_pills = st.sidebar.pills(
                label=i,
                key=f"types_{i}",
                options=["Benefit", "Cost"],
                default="Benefit",
                help="Benefit: The higher, the better. Cost: The lower, the better."
            )
            value = 1 if selected_pills == "Benefit" else -1
            pills_results.append(value)

    weights_form = st.sidebar.radio(
        "METHOD FOR CALCULATING WEIGHTS",
        ["Direct rating", "Pairwise comparison (subjective)", "CRITIC (objective)"], index=0
    )

    if weights_form == "Direct rating":
        metric_values = {}
        for i in metric_name:
            value = st.sidebar.slider(i, min_value=1, max_value=10, value=5, step=1, key=f"metric_slider_{i}")
            metric_values[i] = value
        weights_list = list(metric_values.values())
        weights = np.array(weights_list)
        weights = normalize([weights], norm="l1")
        weights = weights.flatten()
        st.subheader("Normalized weights:")
        result = pd.DataFrame({'Metric': metric_name, 'Weight': np.round(weights, 4)})
        st.write(result)

    elif weights_form == "Pairwise comparison (subjective)":
        st.sidebar.write("## Saaty's pairwise comparison")
        if metrics_num >= 10:
            st.sidebar.write("A higher number of metrics is not recommended for this method.")
        else:
            criteria = data.columns.tolist()
            comparison_matrix = np.ones((metrics_num, metrics_num))
            st.subheader("Calculated weights:")
            for i in range(metrics_num):
                for j in range(metrics_num):
                    if i == j:
                        comparison_matrix[i, j] = 1
                    elif i > j:
                        comparison_matrix[i, j] = eval(
                            st.sidebar.selectbox(f"Compare {criteria[i]} to {criteria[j]}",
                                                 ("1", "3", "5", "7", "9", "1/3", "1/5", "1/7", "1/9")))
                        comparison_matrix[j, i] = 1 / comparison_matrix[i, j]
            weight_derivation = 'geometric'
            weights, rc = ahp_method(comparison_matrix, wd=weight_derivation)
            result = pd.DataFrame({'Metric': metric_name, 'Weight': np.round(weights, 4)})
            st.write(result)

    else:  # CRITIC
        criteria_types = np.ones(data.shape[1])
        weights = critic_weights(data, criteria_types)
        st.subheader("Calculated weights:")
        result = pd.DataFrame({'Metric': metric_name, 'Weight': np.round(weights, 4)})
        st.write(result)

    if st.sidebar.button("Calculating preferences", icon=":material/calculate:", key="filter",
                         width="stretch", type="primary"):

        #data prep
        data_matrix = data.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        scaler = MinMaxScaler()
        data_norm = scaler.fit_transform(data_matrix)
        #protection: 0 dispers.
        data_norm = data_norm + np.random.normal(0, 1e-9, data_norm.shape)

        weights = np.array(weights)
        types_array = np.array(pills_results, dtype=int)

        st.info(f"Data: {data_norm.shape[0]} alternatives, {data_norm.shape[1]} kriterias")
        st.info(f"weights: {np.round(weights, 4)}")
        st.info(f"type of weights: {types_array} (1=Benefit, -1=Cost)")

        #new approach
        def vikor_manual(matrix, weights, types, v=0.5):
            """VIKOR without pymcdm, works with NaN"""
            matrix = np.asarray(matrix, dtype=float)
            weights = np.asarray(weights)
            n_alt, n_crit = matrix.shape
            norm = np.zeros_like(matrix)
            for j in range(n_crit):
                min_j = np.min(matrix[:, j])
                max_j = np.max(matrix[:, j])
                range_j = max_j - min_j
                if range_j < 1e-12:
                    norm[:, j] = 0
                else:
                    if types[j] == 1:
                        norm[:, j] = (matrix[:, j] - min_j) / range_j
                    else:
                        norm[:, j] = (max_j - matrix[:, j]) / range_j
            f_star = np.max(norm, axis=0)
            f_nadir = np.min(norm, axis=0)
            S = np.zeros(n_alt)
            R = np.zeros(n_alt)
            for i in range(n_alt):
                diff = (f_star - norm[i]) / (f_star - f_nadir + 1e-12)
                weighted_diff = weights * diff
                S[i] = np.sum(weighted_diff)
                R[i] = np.max(weighted_diff)
            S_star, S_nadir = np.min(S), np.max(S)
            R_star, R_nadir = np.min(R), np.max(R)
            Q = v * (S - S_star) / (S_nadir - S_star + 1e-12) + (1 - v) * (R - R_star) / (R_nadir - R_star + 1e-12)
            return Q  #lower - better

        def waspas_manual(matrix, weights, types, lambd=0.5):
            """WASPAS: WSM + WPM, higher - better"""
            matrix = np.asarray(matrix, dtype=float)
            weights = np.asarray(weights)
            n_alt, n_crit = matrix.shape
            norm = np.zeros_like(matrix)
            for j in range(n_crit):
                min_j = np.min(matrix[:, j])
                max_j = np.max(matrix[:, j])
                range_j = max_j - min_j
                if range_j < 1e-12:
                    norm[:, j] = 0
                else:
                    if types[j] == 1:
                        norm[:, j] = (matrix[:, j] - min_j) / range_j
                    else:
                        norm[:, j] = (max_j - matrix[:, j]) / range_j
            wsm = np.sum(norm * weights, axis=1)
            wpm = np.zeros(n_alt)
            for i in range(n_alt):
                prod = 1.0
                for j in range(n_crit):
                    prod *= (norm[i, j] ** weights[j])
                wpm[i] = prod
            waspas = lambd * wsm + (1 - lambd) * wpm
            return waspas

        #Methods from lib
        lib_methods = [
            (ARAS(), "ARAS"),
            (CODAS(), "CODAS"),
            (EDAS(), "EDAS"),
            (MABAC(), "MABAC"),
            (MARCOS(), "MARCOS"),
            (PROMETHEE_II('usual'), "PROMETHEE_II"),
            (TOPSIS(), "TOPSIS"),
            (WSM(), "WSM")
        ]
        #methods without lib
        manual_methods = [
            (vikor_manual, "VIKOR", True),   # True -> min
            (waspas_manual, "WASPAS", False) # False -> max
        ]

        prefs = []
        ranks = []
        successful_methods = []

        df_pref = pd.DataFrame()
        df_rank = pd.DataFrame()
        df_pref["Method"] = df.iloc[:, 0]
        df_rank["Method"] = df.iloc[:, 0]

        #lib methods
        for method, name in lib_methods:
            try:
                try:
                    pref = method(data_norm, weights, types_array)
                except TypeError:
                    pref = method(data_norm, weights)
                pref = np.array(pref, dtype=float)
                if np.isnan(pref).any():
                    raise ValueError("NaN")
                #all lib methods -> max
                rank = rankdata(-pref, method='dense')
                prefs.append(pref)
                ranks.append(rank)
                successful_methods.append(name)
                df_pref[f"Pref_{name}"] = pref
                df_rank[name] = rank
                st.success(f"✅ {name} — done")
            except Exception as e:
                st.error(f"❌ {name} failed: {e}")
                with st.expander(f"error {name}"):
                    st.code(traceback.format_exc())

        #methods without lib
        for func, name, minimize in manual_methods:
            try:
                pref = func(data_norm, weights, types_array)
                pref = np.array(pref, dtype=float)
                if np.isnan(pref).any():
                    raise ValueError("NaN")
                if minimize:
                    rank = rankdata(pref, method='dense')
                else:
                    rank = rankdata(-pref, method='dense')
                prefs.append(pref)
                ranks.append(rank)
                successful_methods.append(name)
                df_pref[f"Pref_{name}"] = pref
                df_rank[name] = rank
                st.success(f"✅ {name} (without lib) — done")
            except Exception as e:
                st.error(f"❌ {name} (рwithout lib) failed: {e}")
                with st.expander(f"err {name}"):
                    st.code(traceback.format_exc())

        if len(successful_methods) == 0:
            st.error("successful methods 0")
            st.stop()

        st.success(f"all successful: {len(successful_methods)}")

        #BORDA COUNT
        def borda_count(ranks_list, alt_names):
            n_alt = len(alt_names)
            scores = {name: 0 for name in alt_names}
            for rank_arr in ranks_list:
                for i, alt_name in enumerate(alt_names):
                    scores[alt_name] += (n_alt - rank_arr[i])
            return scores

        borda_scores = borda_count(ranks, alternative_names)
        borda_table = pd.DataFrame(list(borda_scores.items()), columns=['Method', 'Score'])
        borda_table = borda_table.sort_values('Score', ascending=False).reset_index(drop=True)

        st.write("### 🥇 Borda score")
        styled_df = borda_table.style.set_properties(
            subset=pd.IndexSlice[0, :],
            **{'background-color': '#b3a4dd', 'font-weight': 'bold', 'color': 'black'}
        )
        st.dataframe(styled_df)
        st.divider()

        #Rank visual
        st.subheader("Visualisation of the rankings")
        df_rank_success = df_rank[['Method'] + successful_methods]
        df_long = df_rank_success.melt(id_vars='Method', var_name='MCDM Method', value_name='Position in ranking')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_long, x='MCDM Method', y='Position in ranking', hue='Method',
                     palette=pastel_palette, marker='o', markersize=8, linewidth=2, ax=ax)
        plt.title('Position in Ranking Across MCDM Methods', fontsize=16)
        plt.xlabel('MCDM Method', fontsize=12)
        plt.ylabel('Position in ranking', fontsize=12)
        plt.gca().invert_yaxis()
        plt.yticks(range(1, num_alternatives + 1))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Methods')
        plt.tight_layout()
        st.pyplot(fig)

        #heatmap
        if len(ranks) > 1:
            st.divider()
            st.subheader("Visualization of the correlations between rankings")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                corr_matrix = correlation_matrix(np.array(ranks), weighted_spearman)
                plt.figure(figsize=(7, 7))
                visuals.correlation_heatmap(corr_matrix, labels=successful_methods, cmap="Greens")
                fig = plt.gcf()
                st.pyplot(fig)

        st.divider()

        #SENSITIVITY ANALYSIS
        def format_rank_to_string(rank_array, alt_names):
            ranked_alt_names = []
            for r in range(1, len(rank_array) + 1):
                try:
                    alt_index = np.where(rank_array == r)[0][0]
                    ranked_alt_names.append(alt_names[alt_index])
                except IndexError:
                    ranked_alt_names.append(f"ERROR_RANK{r}")
            return ' > '.join(ranked_alt_names)

        #list of methods sensitivity
        sensitivity_methods = {}
        for method, name in lib_methods:
            sensitivity_methods[name] = method
        for func, name, minimize in manual_methods:
            sensitivity_methods[name] = func

        #save info min, max
        minimize_dict = {"VIKOR": True}
        for _, name, minimize in manual_methods:
            minimize_dict[name] = minimize
        for _, name in lib_methods:
            minimize_dict[name] = False  #all lib methods max

        def sensitivity_analysis(data_matrix, original_weights, types_array, alt_names, delta_values=np.linspace(-0.4, 0.4, 20)):
            num_criteria = len(original_weights)
            all_results = []
            for delta in delta_values:
                new_weights_unnorm = np.abs(original_weights + delta)
                sum_w = np.sum(new_weights_unnorm)
                final_weights = new_weights_unnorm / sum_w if sum_w > 1e-6 else np.ones(num_criteria)/num_criteria
                results = {'Weight Change (δ)': f'{delta:+.3f}'}
                for i in range(num_criteria):
                    results[f'C{i+1}(w\'{i+1})'] = f'{final_weights[i]:.4f}'
                for method_name, method_func in sensitivity_methods.items():
                    try:
                        if method_name in ["VIKOR", "WASPAS"]:
                            #methods without lin
                            pref = method_func(data_matrix, final_weights, types_array)
                        else:
                            #methods with lib
                            try:
                                pref = method_func(data_matrix, final_weights, types_array)
                            except TypeError:
                                pref = method_func(data_matrix, final_weights)
                        pref = np.array(pref, dtype=float)
                        if minimize_dict.get(method_name, False):
                            rank = rankdata(pref, method='dense')
                        else:
                            rank = rankdata(-pref, method='dense')
                        rank_string = format_rank_to_string(rank, alt_names)
                        results[f"{method_name} Rank"] = rank_string
                    except Exception as e:
                        results[f"{method_name} Rank"] = f"ERROR: {e}"
                all_results.append(results)
            return pd.DataFrame(all_results)

        temp_alt_names = [f'A{i+1}' for i in range(num_alternatives)]
        df_sensitivity = sensitivity_analysis(data_norm, weights, types_array, temp_alt_names)

        if not df_sensitivity.empty:
            st.header("Sensitivity Analysis")
            st.markdown(f"Analyzed Alternatives: **{', '.join(temp_alt_names)}**")
            column_order = ['Weight Change (δ)'] + [f'C{i+1}(w\'{i+1})' for i in range(metrics_num)] + [f"{m} Rank" for m in successful_methods]
            column_order = [c for c in column_order if c in df_sensitivity.columns]
            df_sensitivity = df_sensitivity[column_order]
            st.dataframe(df_sensitivity, height=35+20*35)

        #% first place
        def evaluate_first_rank_percentage(df_sens, alt_names):
            rank_cols = [col for col in df_sens.columns if ' Rank' in col]
            if not rank_cols:
                return pd.DataFrame()
            df_eval = pd.DataFrame(index=alt_names)
            total_exp = len(df_sens)
            for rank_col in rank_cols:
                df_sens['Winner'] = df_sens[rank_col].apply(lambda x: x.split(' > ')[0] if isinstance(x, str) else '')
                winner_counts = df_sens['Winner'].value_counts()
                winner_pct = (winner_counts / total_exp) * 100
                df_eval[rank_col] = winner_pct
                df_eval = df_eval.fillna(0)
                df_eval[rank_col] = df_eval[rank_col].apply(lambda x: f"{x:.2f} %")
            return df_eval

        df_robustness = evaluate_first_rank_percentage(df_sensitivity, temp_alt_names)
        st.subheader("Percentage of 1st Rank Across Experiments")
        st.dataframe(df_robustness, use_container_width=True)

        #pie chart of wins
        rank_columns = [f"{m} Rank" for m in successful_methods if f"{m} Rank" in df_sensitivity.columns]
        df_all_ranks_long = []
        for _, row in df_sensitivity.iterrows():
            for method_col in rank_columns:
                rank_string = row[method_col]
                if not isinstance(rank_string, str) or 'ERROR' in rank_string:
                    continue
                ordered_alts = rank_string.split(' > ')
                for i, alt_temp in enumerate(temp_alt_names):
                    try:
                        rank_value = ordered_alts.index(alt_temp) + 1
                    except ValueError:
                        continue
                    df_all_ranks_long.append({
                        'Alternative': alt_temp,
                        'Rank': rank_value,
                        'MCDM Method': method_col.replace(' Rank', ''),
                        'Delta': row['Weight Change (δ)']
                    })
        df_ranks_distribution = pd.DataFrame(df_all_ranks_long)

        st.subheader("Pie Chart: Overall Share of 1st Rank Wins")
        actual_alternative_names = methods_name.tolist()
        if len(temp_alt_names) == len(actual_alternative_names):
            legend_map = {temp: actual for temp, actual in zip(temp_alt_names, actual_alternative_names)}
        else:
            legend_map = {}
        df_winners = df_ranks_distribution[df_ranks_distribution['Rank'] == 1].copy()
        if legend_map:
            df_winners['Alternative'] = df_winners['Alternative'].replace(legend_map)
        total_wins = df_winners['Alternative'].value_counts()
        if not total_wins.empty:
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
            ax_pie.pie(total_wins.values, labels=total_wins.index, autopct='%1.1f%%', startangle=90)
            ax_pie.axis('equal')
            ax_pie.set_title('Overall Share of 1st Rank Wins Across All Methods and δ Values', fontsize=10)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig_pie)
        else:
            st.warning("No alternative achieved 1st rank.")

        #graph of rank changes
        def plot_rank_change_sensitivity(df_sens, target_method_col, alt_names, n_alts):
            if target_method_col not in df_sens.columns:
                st.error(f"Column {target_method_col} not found.")
                return
            df_rc = df_sens[['Weight Change (δ)', target_method_col]].copy()
            df_rc['Weight Change (δ)'] = df_rc['Weight Change (δ)'].astype(str).str.replace('+', '', regex=False).astype(float)
            df_rc = df_rc.sort_values('Weight Change (δ)').reset_index(drop=True)
            def get_rank(rank_str, target):
                if not isinstance(rank_str, str):
                    return np.nan
                ordered = rank_str.split(' > ')
                try:
                    return ordered.index(target) + 1
                except ValueError:
                    return np.nan
            for alt in alt_names:
                df_rc[f'Rank_{alt}'] = df_rc[target_method_col].apply(lambda x: get_rank(x, alt))
            df_long = df_rc.melt(id_vars='Weight Change (δ)',
                                 value_vars=[f'Rank_{alt}' for alt in alt_names],
                                 var_name='Alternative', value_name='Rank').dropna(subset=['Rank'])
            df_long['Alternative'] = df_long['Alternative'].str.replace('Rank_', '')
            fig, ax = plt.subplots(figsize=(12, 6))
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=pastel_palette)
            sns.lineplot(data=df_long, x='Weight Change (δ)', y='Rank', hue='Alternative',
                         marker='o', markersize=6, linewidth=2, ax=ax)
            method_clean = target_method_col.replace(' Rank', '')
            ax.set_title(f'Rank Change of Alternatives due to δ ({method_clean})', fontsize=16)
            ax.set_ylabel('Rank', fontsize=12)
            ax.set_xlabel('Additive Weight Change (δ)', fontsize=12)
            ax.set_yticks(np.arange(1, n_alts + 1))
            ax.invert_yaxis()
            ax.grid(axis='both', linestyle='--', alpha=0.5)
            ax.legend(title='Alternatives', loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout()
            st.pyplot(fig)

        for method_name in successful_methods:
            col_name = f"{method_name} Rank"
            if col_name in df_sensitivity.columns:
                with st.expander(f"Rank change of alternatives due to δ for method {method_name}"):
                    plot_rank_change_sensitivity(df_sensitivity, col_name, temp_alt_names, num_alternatives)

        st.info("VIKOR & WASPAS - without lib, other methods from pymcdm.")