n_queries: int = 10
# for idx in range(n_queries):
#     query_idx, query_instance = regressor.query(X)
#     regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))

# y_pred_final, y_std_final=regressor.predict(X_grid.reshape(-1, 1), return_std=True)
# y_pred_final, y_std_final = y_pred_final.ravel(), y_std_final.ravel()
# with plt.style.context("seaborn-white"):
#     plt.figure(figsize=(10, 5))
#     plt.plot(X_grid, y_pred_final)
#     plt.fill_between(X_grid, y_pred_final-y_std_final, y_pred_final+y_std_final, alpha=0.2)
#     plt.scatter(X, y, c='k', s=20)
#     plt.title("final prediction")
#     plt.savefig("ActiveLearning/fig/finalmodel.pdf")