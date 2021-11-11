import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("data/kaggle_survey_2021_responses.csv", header=None, skiprows=2)
columns = ["duration", "age", "gender", "country", "education", "role", "years_exp", "lan_python", "lan_r", "lan_sql",
           "lan_c", "lan_c++", "lan_java", "lan_javascript", "lan_julia", "lan_swift", "lan_bash",
           "lan_matlab", "lan_none", "lan_other",
           "recommend", "ide_jupyter", "ide_rstudio", "ide_visual_studio", "ide_vs_code", "ide_pycharm",
           "ide_spyder", "ide_notepad++", "ide_sublime_text", "ide_vim_emacs", "ide_matlab",
           "ide_jupyter_notebook", "ide_none", "ide_other", "hosted_kaggle", "hosted_colab", "hosted_azure",
           "hosted_paperspace_gradient", "hosted_binder_jupyterhub", "hosted_code_ocean", "hosted_ibm_watson",
           "hosted_amazon_sage_maker", "hosted_amazon_emr", "hosted_google_notebooks", "hosted_google_datalab",
           "hosted_databricks", "hosted_zepl", "hosted_deepnote", "hosted_observable", "hosted_none",
           "hosted_other", "platform", "hardware_nvidia", "hardware_google_tpu", "hardware_aws_trainium",
           "hardware_aws_inferentia", "hardware_none", "hardware_other", "tpu", "visual_matplotlib",
           "visual_seaborn", "visual_plotly", "visual_ggplot", "visual_shiny", "visual_visual_d3_js",
           "visual_altair", "visual_bokeh", "visual_geopoltlib", "visual_leaflet_folium", "visual_none",
           "visual_other", "years_ml", "ml_scikit", "ml_tensor_flow", "ml_keras", "ml_pytorch", "ml_fast_ai",
           "ml_mxnet", "ml_xgboost", "ml_light_gbm", "ml_cat_boost", "ml_prophet", "ml_h2o_3", "ml_caret",
           "ml_tidy_models", "ml_jax", "ml_pytorch_lightning", "ml_hugging_face", "ml_none", "ml_other",
           "algorithm_linear_logistic", "algorithm_tree_forest", "algorithm_gradient_boosting_machines",
           "algorithm_bayesian", "algorithm_evolutionary", "algorithm_dense_neural",
           "algorithm_convolution_neural", "algorithm_generative_adversarial",
           "algorithm_recurrent_neural", "algorithm_transformer_networks", "algorithm_none", "algorithm_other",
           "vision_general_purpose", "vision_image_segmentation", "vision_object_detection",
           "vision_image_classification", "vision_generative_networks", "vision_none", "vision_other",
           "nlp_word_embeddings", "nlp_encoder_decoder", "nlp_contextualized_embeddings", "nlp_transformer_language",
           "nlp_none", "nlp_other", "industry", "company_size", "data_science_individuals", "employer_use_ml",
           "role_analyze_understand", "role_build_run_infrastructure", "role_build_prototypes", "role_build_run_ml",
           "role_improve_ml", "role_research_ml", "role_none", "role_other", "compensation", "spent_five_years",
           "cloud_platforms_aws", "cloud_platforms_azure", "cloud_platforms_google", "cloud_platforms_ibm",
           "cloud_platforms_oracle", "cloud_platforms_sap", "cloud_platforms_sales_force", "cloud_platforms_vmware",
           "cloud_platforms_alibaba", "cloud_platforms_tencent", "cloud_platforms_none", "cloud_platforms_other",
           "best_developer_experience", "cloud_products_amazon_ec2", "cloud_products_microsoft_vm",
           "cloud_products_google_compute_engine", "cloud_products_none", "cloud_products_other",
           "data_storage_azure_data_lake", "data_storage_azure_disk_storage", "data_storage_amazon_s3",
           "data_storage_amazon_elastic_file_system", "data_storage_google_cloud_storage",
           "data_storage_google_cloud_filestore", "data_storage_none", "data_storage_other",
           "ml_products_amazon_sagemaker", "ml_products_azure_machine_learning_studio",
           "ml_products_google_cloud_vertex", "ml_products_data_robot", "ml_products_data_bricks",
           "ml_products_dataiku", "ml_products_alteryx", "ml_products_rapidminer", "ml_products_none",
           "ml_products_other", "big_data_mysql", "big_data_postgresql", "big_data_sqlite",
           "big_data_oracle", "big_data_mongodb", "big_data_snowflake", "big_data_ibm_db2",
           "big_data_microsoft_sql", "big_data_microsoft_azure_sql", "big_data_microsoft_azure_cosmos",
           "big_data_amazon_redshift", "big_data_amazon_aurora", "big_data_amazon_rds",
           "big_data_amazon_dynamo_db", "big_data_google_bigquery", "big_data_google_sql",
           "big_data_google_firestone", "big_data_google_bigtable", "big_data_google_spanner",
           "big_data_none", "big_data_other", "big_data_most_often", "business_intelligence_amazon_quicksight",
           "business_intelligence_microsoft_powerbi", "business_intelligence_google_data_studio",
           "business_intelligence_looker", "business_intelligence_tableau", "business_intelligence_salesforce",
           "business_intelligence_tableau_crm", "business_intelligence_qlik", "business_intelligence_domo",
           "business_intelligence_tibco_spotfire", "business_intelligence_alteryx", "business_intelligence_sisense",
           "business_intelligence_sap", "business_intelligence_microsoft_azure_synapse",
           "business_intelligence_thoughtsplot", "business_intelligence_none", "business_intelligence_other",
           "business_intelligence_most_often", "automl_automated_data_augmentation",
           "automl_automated_feature_engineering", "automl_automated_model_selection",
           "automl_model_architecture_searches", "automl_hyperparemeter_tuning", "automl_automation_full_ml_pipeline",
           "automl_automl_none", "automl_other", "automl_tools_google_cloud_automl",
           "automl_tools_h2o_driverless", "automl_tools_databricks", "automl_tools_datarobot",
           "automl_tools_amazon_sagemaker_autopliot", "automl_tools_azure_automated_ml", "automl_tools_none",
           "automl_tools_other", "ml_experiments_neptune", "ml_experiments_weights_and_biases",
           "ml_experiments_comet", "ml_experiments_sacred_and_omniboard", "ml_experiments_tensorboard",
           "ml_experiments_guild", "ml_experiments_polyaxon", "ml_experiments_clearml",
           "ml_experiments_domino_model_monitor", "ml_experiments_mlflow", "ml_experiments_none",
           "ml_experiments_other", "share_deploy_plotly", "share_deploy_streamlit", "share_deploy_nbviewer",
           "share_deploy_github", "share_deploy_personal_blog", "share_deploy_kaggle", "share_deploy_colab",
           "share_deploy_shiny", "share_deploy_do not_share", "share_deploy_other", "courses_coursera",
           "courses_edx", "courses_kaggle_learning_courses", "courses_datacamp", "courses_fast",
           "courses_udacity", "courses_udemy", "courses_linkedin_learning",
           "courses_cloud_certification_programs", "courses_university", "courses_none", "courses_other",
           "primary_tool_to_analyze", "media_twitter", "media_newsletters", "media_reddit", "media_kaggle",
           "media_course_forums", "media_youtube", "media_podcasts", "media_blogs", "media_journal_publications",
           "media_slack_communities", "media_none", "media_other", "cloud_platform_learn_aws",
           "cloud_platform_learn_microsoft_azure", "cloud_platform_learn_google_cloud","cloud_platform_learn_ibm",
           "cloud_platform_learn_oracle", "cloud_platform_learn_sap", "cloud_platform_learn_vmware",
           "cloud_platform_learn_salesforce", "cloud_platform_learn_alibaba", "cloud_platform_learn_tencent",
           "cloud_platform_learn_none", "cloud_platform_learn_other",
           "cloud_products_learn_amazon_ec2", "cloud_products_learn_microsoft_azure_vm",
           "cloud_products_learn_google_cloud_compute_engine", "cloud_products_learn_none",
           "cloud_products_learn_other", "data_storage_learn_microsoft_azure_data_lake",
           "data_storage_learn_microsoft_azure_disk", "data_storage_learn_amazon_s3",
           "data_storage_learn_amazon_elastic_file_system", "data_storage_learn_google_cloud_storage",
           "data_storage_learn_google_cloud_filestore", "data_storage_learn_none", "data_storage_learn_other",
           "ml_products_learn_amazon_sagemaker", "ml_products_learn_azure_ml_studio",
           "ml_products_learn_google_cloud_vertex", "ml_products_learn_datarobots", "ml_products_learn_databricks",
           "ml_products_learn_dataiku", "ml_products_learn_alteryx", "ml_products_learn_rapidminer",
           "ml_products_learn_none", "ml_products_learn_other", "big_data_learn_mysql", "big_data_learn_postgresql",
           "big_data_learn_sqlite", "big_data_learn_oracle", "big_data_learn_mongobd", "big_data_learn_snowflake",
           "big_data_learn_ibm_db2", "big_data_learn_microsoft_sql", "big_data_learn_microsoft_azure_sql",
           "big_data_learn_microsoft_azure_cosmos", "big_data_learn_amazon_redshift",
           "big_data_learn_amazon_aurora", "big_data_learn_amazon_rds", "big_data_learn_amazon_dynamodb",
           "big_data_learn_google_bigquery", "big_data_learn_google_sql", "big_data_learn_google_firestore",
           "big_data_learn_google_bigtable", "big_data_learn_google_spanner", "big_data_learn_none",
           "big_data_learn_other", "business_intelligence_learn_amazon_quicksight",
           "business_intelligence_learn_microsoft_powerbi", "business_intelligence_learn_google_data_studio",
           "business_intelligence_learn_looker", "business_intelligence_learn_tableau",
           "business_intelligence_learn_salesforce", "business_intelligence_learn_einstein_analytics",
           "business_intelligence_learn_qlik", "business_intelligence_learn_domo",
           "business_intelligence_learn_tibco_spotfire", "business_intelligence_learn_alteryx",
           "business_intelligence_learn_sisense", "business_intelligence_learn_sap",
           "business_intelligence_learn_microsoft_azure_synapse", "business_intelligence_learn_thoughtsplot",
           "business_intelligence_learn_none", "business_intelligence_learn_other",
           "auto_ml_learn_automated_data_augmentation", "auto_ml_learn_automated_feature_engineering",
           "auto_ml_learn_automated_model_selection", "auto_ml_learn_automated_model_architecture",
           "auto_ml_learn_automated_hyperparameter_tuning", "auto_ml_learn_automation_full_pipelines",
           "auto_ml_learn_none", "auto_ml_learn_other", "auto_ml_tool_learn_google_cloud",
           "auto_ml_tool_learn_h2o_driverless", "auto_ml_tool_learn_databricks", "auto_ml_tool_learn_datarobot",
           "auto_ml_tool_learn_amazon_sagemaker", "auto_ml_tool_learn_azure_automated_ml",
           "auto_ml_tool_learn_none", "auto_ml_tool_learn_other", "ml_experiments_learn_neptune",
           "ml_experiments_learn_weights_and_biases", "ml_experiments_learn_comet",
           "ml_experiments_learn_sacred_and_omniboard", "ml_experiments_learn_tensorboard",
           "ml_experiments_learn_guild", "ml_experiments_learn_polyaxon", "ml_experiments_learn_trains",
           "ml_experiments_learn_domino_model_monitor", "ml_experiments_learn_mlflow",
           "ml_experiments_learn_none", "ml_experiments_learn_other"
           ]
df.columns = columns

df.head()

#df["age"] = df["age"].astype("category")
#df["age"].cat.reorder_categories(["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
 #                                 "55-59", "60-69", "70+"])

# let us look at the distribution of ages first
df["age"].value_counts().plot(kind="bar")
# we see that most are on the younger side. Let us now try to see the languages popular within each age group

# create an id for each row
id = range(len(df))
df["id"] = id
# save the columns I want to keep
to_keep = ["id", "age", "lan_python", "lan_r", "lan_sql",
           "lan_c", "lan_c++", "lan_java", "lan_javascript", "lan_julia", "lan_swift", "lan_bash",
           "lan_matlab", "lan_none", "lan_other"]
# save the names of the wide columns that should be transofmred to long
languages = ["lan_python", "lan_r", "lan_sql",
           "lan_c", "lan_c++", "lan_java", "lan_javascript", "lan_julia", "lan_swift", "lan_bash",
           "lan_matlab", "lan_none", "lan_other"]
# save a copy of the data frame that includes only the columns i want to keep
hold = df[to_keep]
# if a user uses a particular language return true, otherwise false
hold[languages] = hold[languages].applymap(lambda x: False if x is np.nan else True)
hold.head()
# now convert to long
hold = pd.melt(hold, id_vars=["id", "age"], value_vars=languages, var_name="language", value_name="used")
# remove rows where used is false
hold = hold[hold["used"]==True]
# remove the "lan_" part from the language name
hold["language"] = hold["language"].str[4:]
# we can now generate the stacked bar graph
# first calculate the size of each group
hold = hold.groupby(["age", "language"]).size().reset_index(name="counts")
# now pivot the table so that we have a column for each language
hold = hold.pivot_table(index="age", columns="language", values="counts")
hold.plot(kind="bar", stacked=True)
# it would be useful to look at the percentages within each age group
hold["total"] = hold.sum(axis=1)
for column in hold.columns:
    hold[column] = hold[column] / hold["total"] * 100
hold.drop("total", axis=1, inplace=True)
hold.plot(kind="bar", stacked=True, color=plt.cm.Paired(np.arange(len(hold.columns))))
plt.legend(ncol=4, loc="upper left")

