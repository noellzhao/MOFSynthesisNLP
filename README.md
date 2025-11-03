# MOFSynthesisNLP
Codebase for natural language processing (NLP) workflows used in extracting synthesis information from Metal–Organic Framework (MOF) research literature.

## Overview

This repository contains the code and workflow developed to identify and extract MOF synthesis procedures from scientific publications.

## Workflow Summary

The overall process includes the following steps:

1. **Manual annotation processing** – curate and normalize human-labeled synthesis paragraphs.
2. **Publication data preprocessing** – extract text components (e.g., abstracts, paragraphs, captions) from publication XMLs.
3. **GPT-based classification** – use GPT models to identify paragraphs describing MOF synthesis procedures.
4. **Structured synthesis extraction** – convert identified synthesis text into a structured, machine-readable format.
5. **Model fine-tuning** – fine-tune a BERT-based classifier using GPT-generated labels to automate synthesis paragraph detection.

## Related Literature

1. Zhao, Xintong, Jair Fernando Fajardo Rojas, Jacob Furst, Katherine Ardila, Kyle Langlois, Yuan An, Xiaohua Hu, Fernando Uribe-Romo, Diego Gomez-Gualdron, and Jane Greenberg. "Expert-Guided LLM Approach for Sequence-Aware Extraction of MOF Synthesis." (2025). DOI[https://doi.org/10.26434/chemrxiv-2025-x90hc]
2. Zhao, Xintong. "Large-scale materials knowledge extraction using LLMs and human-in-the-loop." PhD diss., Drexel University, 2025. DOI[https://doi.org/10.17918/00010968]

## Model
A Fine-tuned Bert Model for MOF Synthesis Classification (https://huggingface.co/noellzhao/MOF_SynthesisDetection)
