import numpy as np
import click
import pandas as pd


def monte_carlo_tdf(
    start_age: int,
    end_age: int,
    expected_returns: list[float],
    risks: list[float],
    initial_investment: float = 1.0,
    n_simulations: int = 10000,
    seed: int = 42,
):
    """
    ターゲットデートファンドのモンテカルロシミュレーション
    各年齢ごとに期待リターン・リスク（標準偏差）が与えられる場合、
    対数正規分布を仮定して資産推移をシミュレーションする。
    """
    np.random.seed(seed)
    n_years = end_age - start_age + 1
    assert len(expected_returns) == n_years
    assert len(risks) == n_years

    # 各年齢ごとにリターンを生成
    results = np.zeros((n_simulations, n_years + 1))
    results[:, 0] = initial_investment
    for i in range(n_years):
        mu = expected_returns[i]
        sigma = risks[i]
        # 対数正規分布: log(1+リターン)が正規分布
        annual_log_return = np.random.normal(
            loc=np.log(1 + mu), scale=sigma, size=n_simulations
        )
        annual_return = np.exp(annual_log_return) - 1
        results[:, i + 1] = results[:, i] * (1 + annual_return)
    return results


@click.command()
@click.option('--csv', type=click.Path(exists=True), help='年齢ごとのリターン・リスク・積立金額のCSVファイル（age,expected_return,risk,contribution列）')
@click.option('--start-age', type=int, help='開始年齢（CSV未指定時は必須）')
@click.option('--end-age', type=int, help='終了年齢（CSV未指定時は必須）')
@click.option('--expected-returns', type=float, multiple=True, help='各年齢ごとの期待リターン（複数指定可、CSV未指定時のみ）')
@click.option('--risks', type=float, multiple=True, help='各年齢ごとのリスク（標準偏差, 複数指定可、CSV未指定時のみ）')
@click.option('--contributions', type=float, multiple=True, help='各年齢ごとの年間積立金額（複数指定可、CSV未指定時のみ）')
@click.option('--initial-investment', type=float, default=None, show_default=True, help='開始時点の積立金の月額（デフォルトは最初の年の積立金額/12）')
@click.option('--n-simulations', type=int, default=10000, show_default=True, help='シミュレーション回数')
@click.option('--seed', type=int, default=42, show_default=True, help='乱数シード')
def main(csv, start_age, end_age, expected_returns, risks, contributions, initial_investment, n_simulations, seed):
    """
    ターゲットデートファンドのモンテカルロシミュレーション（月次・年毎の積立対応）
    """
    if csv:
        df = pd.read_csv(csv)
        if not set(['age', 'expected_return', 'risk', 'contribution']).issubset(df.columns):
            raise ValueError('CSVにはage,expected_return,risk,contribution列が必要です')
        df = df.sort_values('age')
        start_age = int(df['age'].iloc[0])
        end_age = int(df['age'].iloc[-1])
        expected_returns = df['expected_return'].tolist()
        risks = df['risk'].tolist()
        contributions = df['contribution'].tolist()
    else:
        if start_age is None or end_age is None:
            raise ValueError('CSV未指定時は--start-age, --end-ageが必要です')
        expected_returns = list(expected_returns)
        risks = list(risks)
        contributions = list(contributions)
        n_years = end_age - start_age + 1
        if len(expected_returns) != n_years or len(risks) != n_years or len(contributions) != n_years:
            raise ValueError(f"期待リターン・リスク・積立金額の数は年数（{n_years}）と一致させてください")
    if initial_investment is None:
        initial_investment = contributions[0] / 12
    results = monte_carlo_tdf_monthly(
        start_age=start_age,
        end_age=end_age,
        expected_returns=expected_returns,
        risks=risks,
        contributions=contributions,
        initial_investment=initial_investment,
        n_simulations=n_simulations,
        seed=seed,
    )
    # 結果のサマリーを表示
    final_assets = results[:, -1]
    print(f"\n{n_simulations}回シミュレーションの最終資産サマリー:")
    print(f"平均: {np.mean(final_assets):.2f}")
    print(f"中央値: {np.median(final_assets):.2f}")
    print(f"標準偏差: {np.std(final_assets):.2f}")
    print(f"5パーセンタイル: {np.percentile(final_assets, 5):.2f}")
    print(f"95パーセンタイル: {np.percentile(final_assets, 95):.2f}")


def monte_carlo_tdf_monthly(
    start_age: int,
    end_age: int,
    expected_returns: list[float],
    risks: list[float],
    contributions: list[float],
    initial_investment: float,
    n_simulations: int = 10000,
    seed: int = 42,
):
    """
    月次でターゲットデートファンドのモンテカルロシミュレーション
    各年齢ごとに期待リターン・リスク・積立金額（年単位）が与えられる場合、
    それぞれを12ヶ月分に展開して月次シミュレーションを行う。
    initial_investmentは開始時点の積立金の月額（初月の積立金額）とする。
    """
    np.random.seed(seed)
    n_years = end_age - start_age + 1
    n_months = n_years * 12
    # 年ごとのパラメータを月ごとに展開
    monthly_returns = np.repeat(expected_returns, 12)
    monthly_risks = np.repeat(risks, 12)
    monthly_contributions = np.repeat([c / 12 for c in contributions], 12)
    assert len(monthly_returns) == n_months
    assert len(monthly_risks) == n_months
    assert len(monthly_contributions) == n_months
    results = np.zeros((n_simulations, n_months + 1))
    results[:, 0] = initial_investment
    for i in range(n_months):
        mu_annual = monthly_returns[i]
        sigma_annual = monthly_risks[i]
        # 年次リターン・リスクを月次に変換
        mu_monthly = np.log(1 + mu_annual) / 12
        sigma_monthly = sigma_annual / np.sqrt(12)
        monthly_log_return = np.random.normal(
            loc=mu_monthly, scale=sigma_monthly, size=n_simulations
        )
        monthly_return = np.exp(monthly_log_return) - 1
        # 積立金額を加算してからリターンを適用
        results[:, i + 1] = (results[:, i] + monthly_contributions[i]) * (1 + monthly_return)
    return results


if __name__ == "__main__":
    main()
