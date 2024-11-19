# %%
import streamlit as st
import pandas as pd
import numpy as np 
import plotly.express as px
from dataclasses import dataclass
from scipy.stats import bernoulli, binom, beta
from scipy.special import betaln
import math
from py_sample_size.sample_size_calculator import noninferiority_test, superiority_test, min_sample_size
from statsmodels.stats.proportion import proportions_ztest


# %%
@dataclass
class BetaData:
    alpha: int
    beta: int

    def update_params(self, a, b):
        self.alpha += a
        self.beta += b

def prob_B_beats_A(a_alpha, a_beta, b_alpha, b_beta):
    total = 0
    for i in range(b_alpha):
        total += np.exp(betaln(a_alpha + i, a_beta + b_beta) - math.log(b_beta + i) - betaln(i+1, b_beta) - betaln(a_alpha, a_beta))
    return total

# assumes loss from choosing B when B is actually worse
def loss(a_alpha, a_beta, b_alpha, b_beta):
    l = (math.exp(betaln(b_alpha + 1, b_beta) - betaln(b_alpha, b_beta)) * prob_B_beats_A(a_alpha, a_beta, b_alpha + 1, b_beta)) - \
    (math.exp(betaln(a_alpha + 1, a_beta) - betaln(a_alpha, a_beta)) * prob_B_beats_A(a_alpha + 1, a_beta, b_alpha, b_beta))

    return l

def simulate_experiment(
    priors: list[BetaData],
    true_rates: list[float],
    obs_per_step: int = 100,
    max_timesteps: int = 100,
    minimum_loss: float = 0.001
):
    # initialize some stuff
    treatment_dist = priors[0]
    control_dist = priors[1]
    did_someone_win = False
    i = 1
    exp_data = {'timestep': [], 'treat_obs': [], 'treat_succ': [], 'control_obs': [], 'control_succ': [], 'treat_loss': [], 'control_loss': []}
    winner = None

    #loop through timesteps
    while i < max_timesteps and not did_someone_win:
        treat_succ = bernoulli.rvs(true_rates[0], size=obs_per_step).sum()
        control_succ = bernoulli.rvs(true_rates[1], size=obs_per_step).sum()

        #update priors
        treatment_dist.update_params(treat_succ, obs_per_step - treat_succ)
        control_dist.update_params(control_succ, obs_per_step - control_succ)

        #calculate loss
        exp_data['control_loss'].append(loss(control_dist.alpha, control_dist.beta, treatment_dist.alpha, treatment_dist.beta))
        exp_data['treat_loss'].append(loss(treatment_dist.alpha, treatment_dist.beta, control_dist.alpha, control_dist.beta))
        exp_data['timestep'].append(i)
        exp_data['treat_obs'].append(obs_per_step)
        exp_data['treat_succ'].append(treat_succ)
        exp_data['control_obs'].append(obs_per_step)
        exp_data['control_succ'].append(control_succ)

        #check if someone won
        if exp_data['treat_loss'][-1] < minimum_loss and exp_data['control_loss'][-1] < minimum_loss:
            winner = 'Tie'
            did_someone_win = True
        elif exp_data['treat_loss'][-1] < minimum_loss:
            winner = 'Treatment'
            did_someone_win = True
        elif exp_data['control_loss'][-1] < minimum_loss:
            winner = 'Control'
            did_someone_win = True
        
        i+=1

    return pd.DataFrame(exp_data), winner

# %%
def many_experiments(
    num_experiments: int,
    priors: list[BetaData],
    true_rates: list[float],
    obs_per_step: int = 100,
    max_timesteps: int = 100,
    minimum_loss: float = 0.001,
):
    lengths = []
    winner_dict = {}
    all_losses = pd.DataFrame(columns=['treat_loss', 'control_loss', 'timestep', 'sim_number'])
    if true_rates[0] > true_rates[1]:
        true_winner = 'Treatment'
    else:
        true_winner = 'Control'

    for i in range(num_experiments):
        prior_copy = [BetaData(p.alpha, p.beta) for p in priors]
        results, winner = simulate_experiment(prior_copy, true_rates, obs_per_step, max_timesteps, minimum_loss)
        
        losses = results[['treat_loss', 'control_loss', 'timestep']]
        losses.loc[:,'sim_number'] = i

        all_losses = pd.concat([all_losses, losses], ignore_index=True, axis=0)

        exp_length = len(results)
        lengths.append(exp_length)

        if winner in winner_dict:
            winner_dict[winner] += 1
        else:
            winner_dict[winner] = 1

        my_bar.progress(i/num_experiments, text=f'Working on Simulation {i+1} of {num_experiments}')
    
    accuracy = winner_dict[true_winner] / num_experiments


    return np.array(lengths), accuracy, all_losses


# %%
st.title("Square Experiment Tool")


params = st.container()
test_type = None
with params:
    st.markdown('#### Design Parameters')
    with st.container(border=True):
        des1, des2, des3= st.columns(3)
        with des1:
            metric_type = st.radio('Discrete or continuous metric?', ['Discrete', 'Continuous'])
        with des2:
            sides = st.radio('1-Sided or 2-Sided?', ['1-Sided', '2-Sided'])
            
        with des3: 
            if sides == '1-Sided':
                test_type = st.radio('What kind of 1-Sided test?',['Superiority', 'Non-Inferiority'])

        

        if metric_type == 'Continuous':
            st.caption('Note: We currently only support power analyses when the variance is the same in the test and control group')


        param1, param2, param3, param4 = st.columns(4)
        with param1:
            if metric_type == 'Discrete':
                base_rate = st.number_input('Base Rate (%)', min_value=0.0, max_value=100.0, value=1.0, step=.1, format='%f') / 100
            else:
                base_val = st.number_input('Base Value', min_value=0.0, value=1.0, step=.01, format='%f')
        with param2:
            mde = st.number_input('Relative MDE (%)', min_value=0.0, max_value=100.0, value=10.0, step=1.0, format='%f') / 100
        with param3:
            fpr = st.number_input('FPR (%)', min_value=0, max_value=100, value=5, step=1, format='%i') / 100
        with param4:
            power = st.number_input('Power (%)', min_value=0, max_value=100, value=80, step=1, format='%i') / 100

        if metric_type == 'Continuous':
            base_var = st.number_input('Base Variance', min_value=0.0, value=1.0, step=1.0, format='%f')
        if test_type == 'Non-Inferiority' and sides == '1-Sided':
            margin = st.number_input('Non-Inferiority Margin (%)', min_value=0, max_value=100, value=0, step=1, format='%i') / 100

    st.markdown('#### Length Parameters')
    with st.container(border=True):
        obs_per_step = st.number_input('Expected Total Number of Observations Per Day', min_value=1, max_value=1000000, value=1000, step=1, format='%i')
        min_length = st.number_input('Minimum Length of Experiment (Days)', min_value=1, max_value=1000, value=7, step=1, format='%i')
        cure_time = st.number_input('Extra Time for Data Curing (Days)', min_value=0, max_value=1000, value=7, step=1, format='%i')

    with st.expander('Advanced Parameters (Optional)', expanded=False):
        n_test_variants = st.number_input('Number of Test Variants', min_value=1, max_value=10, value=1, step=1, format='%i')

        testing_split = st.slider('Testing Split (%)',
                                            min_value=1,
                                            max_value=100,
                                            value=50,
                                            step=1,
                                            format='%i') / 100

        multiple_testing_correction = st.selectbox('Multiple Testing Correction', 
                                                ['Bonferroni', 'Holm-Sidak', 'Benjamini-Hochberg', None],
                                                index=3)
        

    st.page_link(page="https://www.notion.so/square-seller/Experiment-Selection-Approach-489532d77497450fbbf28ecb3fe508be?pvs=4",
                    label="***Need Help Selecting Parameters?   Click here to read Square Data's Experiment Guide***",
                    icon=':material/help_outline:',
                    help='Notion Page on Experiment Selection & Approach',
                    use_container_width=True)    

if metric_type == 'Discrete':
    bayes = st.toggle("Compare to a Bayesian Approach?")
    if bayes:
        st.markdown('#### Bayesian Parameters')    
        with st.container():

            with st.container(border = True):
                max_timesteps = st.number_input('Maximum Number of Timesteps', min_value=1, max_value=1000, value=100, step=1, format='%i')
                num_experiments = st.number_input('Number of Simulations', min_value=1, max_value=1000, value=100, step=1, format='%i')

            with st.container(border=True):
                prior_alpha_col, prior_beta_col = st.columns(2)
                with prior_alpha_col:
                    treat_alpha = st.number_input('Treatment Alpha', min_value=0, max_value=10000, value=1, step=1, format='%i')
                    cont_alpha = st.number_input('Control Alpha', min_value=0, max_value=10000, value=1, step=1, format='%i')
                    
                with prior_beta_col:
                    treat_beta = st.number_input('Treatment Beta', min_value=0, max_value=10000, value=1, step=1, format='%i')
                    cont_beta = st.number_input('Control Beta', min_value=0, max_value=10000, value=1, step=1, format='%i')

                treatment_dist = BetaData(treat_alpha, treat_beta)
                control_dist = BetaData(cont_alpha, cont_beta)

            with st.container(border=True):
                minimum_loss = st.number_input('Minimum Loss Threshold', min_value=0.0, max_value=1.0, value=0.001, step=0.001, format='%f')
                st.caption("This tool uses Expected Loss as the stopping criterion")


with st.form('Sample Size Calculator'):
    submitted = st.form_submit_button('Calculate Sample Size')
    if submitted:
        with st.spinner('Calculating Sample Size...'):

            # calculate the frequentist sample size
            if metric_type == 'Discrete':
                if sides == '2-Sided':
                    control_samples, test_samples = min_sample_size(
                        baseline_rate = base_rate,
                        mde = mde,
                        margin = 0,
                        desired_fpr = fpr,
                        desired_power = power,
                        test_split = testing_split,
                        test_type = 'two-sided',
                        n_test_variants = n_test_variants,
                        mde_positive = True,
                        multiple_testing_correction = multiple_testing_correction,
                        is_binary = True
                    )
                elif test_type == 'Superiority':
                    control_samples, test_samples = min_sample_size(
                        baseline_rate = base_rate,
                        mde = mde,
                        margin = 0,
                        desired_fpr = fpr,
                        desired_power = power,
                        test_split = testing_split,
                        test_type = 'superiority',
                        n_test_variants = n_test_variants,
                        mde_positive = True,
                        multiple_testing_correction = multiple_testing_correction,
                        is_binary = True
                    )
                elif test_type == 'Non-Inferiority':
                    control_samples, test_samples = min_sample_size(
                        baseline_rate = base_rate,
                        mde = mde,
                        margin = -(margin),
                        desired_fpr = fpr,
                        desired_power = power,
                        test_split = testing_split,
                        test_type = 'non-inferiority',
                        n_test_variants = n_test_variants,
                        mde_positive = True,
                        multiple_testing_correction = multiple_testing_correction,
                        is_binary = True
                    )

            elif metric_type == 'Continuous':
                if sides == '2-Sided':
                    control_samples, test_samples = min_sample_size(
                        baseline_rate = base_val,
                        mde = mde,
                        margin = 0,
                        desired_fpr = fpr,
                        desired_power = power,
                        test_split = testing_split,
                        test_type = 'two-sided',
                        baseline_variance = base_var,
                        n_test_variants = n_test_variants,
                        mde_positive = True,
                        multiple_testing_correction = multiple_testing_correction,
                        is_binary = False
                    )
                elif test_type == 'Superiority':
                    control_samples, test_samples = min_sample_size(
                        baseline_rate = base_val,
                        mde = mde,
                        margin = 0,
                        desired_fpr = fpr,
                        desired_power = power,
                        test_split = testing_split,
                        test_type = 'superiority',
                        baseline_variance = base_var,
                        n_test_variants = n_test_variants,
                        mde_positive = True,
                        multiple_testing_correction = multiple_testing_correction,
                        is_binary = False
                    )
                elif test_type == 'Non-Inferiority':
                    control_samples, test_samples = min_sample_size(
                        baseline_rate = base_val,
                        mde = mde,
                        margin = -(margin),
                        desired_fpr = fpr,
                        desired_power = power,
                        test_split = testing_split,
                        test_type = 'non-inferiority',
                        baseline_variance = base_var,
                        n_test_variants = n_test_variants,
                        mde_positive = True,
                        multiple_testing_correction = multiple_testing_correction,
                        is_binary = False
                    )

        if n_test_variants > 1:
            test_samples = round(test_samples / n_test_variants)

        out1, out2, out3 = st.columns(3)

        exp_length = math.ceil((control_samples + test_samples) / obs_per_step)
        if exp_length < min_length:
            st.warning(f"The calculated sample size is less than the number of observations that can be observed in {min_length} days. \
                        Setting total sample size to ({min_length} days) * ({obs_per_step} observations/day) = {min_length * obs_per_step}")
            
            control_samples = round(min_length * obs_per_step * (1 - testing_split))
            test_samples = round(min_length * obs_per_step * testing_split / n_test_variants)

        with out1:
            st.metric(label='Control Group Sample Size', value=control_samples)
        with out2:
            if n_test_variants > 1:
                st.metric(label='Test Sample Size (per variant)', value=test_samples)
            else:
                st.metric(label='Test Group Sample Size', value=test_samples)
        with out3:
            st.metric(label='Total Experiment Length', value=f"{max(exp_length, min_length) + cure_time} days")

        st.caption('Note that since the calculator is based on a simulation, the sample size may vary slightly between runs.')


with st.form('Run Simulation'):
    bayes_submitted = st.form_submit_button('Run Simulation')
    if bayes_submitted:
        
        # run the bayesian simulations
        my_bar = st.progress(0, text='Running Simulations')

        sims, acc, all_losses = many_experiments(num_experiments=num_experiments, priors=[treatment_dist, control_dist], true_rates=[base_rate * (1+mde), base_rate], obs_per_step=round(obs_per_step), max_timesteps=max_timesteps, minimum_loss=minimum_loss)
        median_length = np.median(sims)
        pct_90_length = np.percentile(sims, 90)
        max_length = np.max(sims)
        min_length = np.min(sims)

        st.subheader('Typical Length of a Bayesian Experiment')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Minimum', f'{round(min_length)} days')
        col2.metric('Median', f'{round(median_length)} days')
        col3.metric('90th Percentile', f'{round(pct_90_length)} days')
        col4.metric('Max', f'{round(max_length)} days')
        
        fig = px.ecdf(x=sims, ecdfnorm='percent', title='Distribution of Experiment Lengths')
        st.plotly_chart(fig)

        st.subheader('Accuracy of Bayesian Method')
        st.metric('Accuracy', f'{round(acc * 100, 2)}%')

        my_bar.empty()




st.page_link(page="https://github.com/squareup/py-sample-size",
            label="Sample Size Package Documentation",
            help='Link to Square\'s Internal Sample Size Package py-sample-size')
