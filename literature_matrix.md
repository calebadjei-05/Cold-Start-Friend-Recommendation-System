# Literature Matrix

Add at least 3 papers for the proposal and at least 6 papers for the final hackathon report.

| Paper citation | Research problem | Data/source | Method | Evaluation | Limitation | How it informs our project |
| --- | --- | --- | --- | --- | --- | --- |
| **Yu, J. (2024)** | Friend recommendation at scale; retrieval, ranking, cold-start | Social network platforms (FoF, contact-book, embeddings) | Two-stage retrieval + ranking architecture | Precision at scale across retrieval sources | Cold-start strategies from item recommendation don't transfer well | Establishes the base architecture our project extends |
| **Ureña-Carrion, Saramäki & Kivelä (2020)** | Estimating tie strength from communication patterns | Temporal communication data | Temporal feature analysis (stability, burstiness, active days) | Compared against simple contact-count baselines | Not cold-start specific | Basis for our frequency/recency/inter-event score |
| **Rosenfeld et al. (2018)** | Predicting relationship strength without reading message content | WhatsApp activity metadata | Metadata-only prediction | Useful predictions from metadata alone | Single-platform study | Validates our privacy-conscious, metadata-only design |
| **Ma, Li & Zhou (2024), *Tail-STEAK*** | Label and neighborhood sparsity for tail/cold-start users | Sparse recommendation datasets | Tail-aware recommendation framework | Performance on sparse/tail users vs. standard models | Item recommendation, not friend recommendation | Shows cold-start/tail-user problem is still unsolved |
| **Sankar et al. (2021)** | GNN-based ranking; structural vs. temporal signals | Graph-structured recommendation data | Graph Neural Network (GNN) ranking | Ranking effectiveness on graph data | Relies on structure, misses temporal signal | Justifies our temporal score as a complementary signal |



### Related work
1. Yu, J. (2024). "Lessons Learnt from Building Friend Recommendation Systems." In Proceedings of the 17th ACM International Conference on Web Search and Data Mining (WSDM '24), Merida, Mexico. — establishes the two-stage retrieval-and-ranking architecture and the cold-start problem; supports our overall system design.

2. Ureña-Carrion, J., Saramäki, J., & Kivelä, M. (2020). "Estimating tie strength in social networks using temporal communication data." EPJ Data Science, 9(1), 37. — shows temporal communication features outperform simple contact counts for tie strength; supports our composite score.

3. Rosenfeld, A., et al. (2018). "A Study of WhatsApp Usage Patterns and Prediction Models without Message Content." arXiv preprint arXiv:1802.03393. — shows useful predictions from WhatsApp metadata alone, without message content; supports our privacy-conscious, metadata-only approach.

4. Ma, Y., Li, J., & Zhou, K. (2024). "Tail-STEAK." In Proceedings of the AAAI Conference on Artificial Intelligence, 38(8), 8877–8885. — identifies label and neighborhood sparsity as key challenges for tail/cold-start users; supports the relevance of our cold-start focus.

5. Sankar, A., et al. (2021). "Graph Neural Networks for Friend Ranking in Large-scale Social Platforms." In Proceedings of the Web Conference 2021 (WWW '21), Ljubljana, Slovenia. — shows GNN ranking is effective but relies mainly on graph structure rather than temporal signal; positions our score as complementary.

