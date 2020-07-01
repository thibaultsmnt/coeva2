success_rates = calculate_success_rates(
        efficient_results, encoder, THRESHOLD, model
    )
    success_rate_df = pd.DataFrame(
        success_rates, columns=["o{}".format(i + 1) for i in success_rates.shape[1]]
    )
    success_rate_df.to_csv()