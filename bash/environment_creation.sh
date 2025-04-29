#!/bin/bash

echo "BOLT Environment creation..."
conda env remove --name bolt_env
conda create --name bolt_env --file ./requirements/requirements_bolt_env.txt
echo "BOLT Environment created."

echo "CAT PAW Environment creation..."
conda env remove --name catpaw_env
conda create --name catpaw_env --file ./requirements/requirements_catpaw_env.txt
echo "CAT PAW Environment created."

echo "LLMs Environment creation..."
conda env remove --name llm_env
conda create --name llm_env --file ./requirements/requirements_llm_env.txt
echo "LLMs Environment created."

echo "Evaluation Environment creation..."
conda env remove --name csctg_eval_env
conda create --name csctg_eval_env --file ./requirements/requirements_csctg_eval_env.txt
echo "Evaluation Environment created."

echo "CTG BART Environment creation..."
conda env remove --name ctg_bart_env
conda create --name ctg_bart_env --file ./requirements/requirements_ctg_bart_env.txt
echo "CTG BART Environment created."

echo "CTRL Environment creation..."
conda env remove --name ctrl_env
conda create --name ctrl_env --file ./requirements/requirements_ctrl_env.txt
echo "CTRL Environment created."

echo "DExperts Environment creation..."
conda env remove --name dexperts_env
conda create --name dexperts_env --file ./requirements/requirements_dexperts_env.txt
echo "DExperts Environment created."

echo "DisCup Environment creation..."
conda env remove --name discup_env
conda create --name discup_env --file ./requirements/requirements_discup_env.txt
echo "DisCup Environment created."

echo "GeDi Environment creation..."
conda env remove --name gedi_env
conda create --name gedi_env --file ./requirements/requirements_gedi_env.txt
echo "GeDi Environment created."

echo "Multi CTG Environment creation..."
conda env remove --name multi_ctg_env
conda create --name multi_ctg_env --file ./requirements/requirements_multi_ctg_env.txt
echo "Multi CTG Environment created."

echo "Prior CTG Environment creation..."
conda env remove --name prior_ctg_env
conda create --name prior_ctg_env --file ./requirements/requirements_prior_ctg_env.txt
echo "Prior CTG Environment created."

