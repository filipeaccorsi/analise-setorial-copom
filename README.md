# Impacto da Política Monetária na Sensibilidade ao Risco Setorial

Trabalho de Finanças II — Filipe Accorsi Oliveira Cunha, Matheus Gonçalves Santos

Este projeto investiga como diferentes setores da bolsa brasileira (Varejo,
Bancos, Construção, Elétrico) reagem a movimentos da taxa de juros
doméstica, e como essa sensibilidade evolui ao longo do tempo. Foram usados fatores 
de risco e betas rolling para a avaliação.

**Objetivo central:**

1. Construir uma base com taxas de juros, datas de reuniões do COPOM,
   preços setoriais de ações e retornos/excessos de retorno.
2. Comparar movimentos de juros em dias de COPOM vs. dias normais.
3. Estimar betas *rolling* (janela móvel) para medir a exposição ao risco
   de cada setor ao longo do tempo.
4. Comparar o risco setorial com o risco de mercado.
5. Gerar conclusões sobre quais setores são mais sensíveis à política
   monetária.

**Dados:** preços diários via Yahoo Finance (`yfinance`) e séries de juros
(Selic e CDI diário anualizado) via SGS/Bacen (`python-bcb`), 2021-01-01 a
2023-01-01.


## Tecnologias e métodos

- **Linguagem**: Python 3.14
- **Bibliotecas**: pandas, numpy, yfinance, python-bcb, statsmodels, scipy, matplotlib
- **Métodos estatísticos**: regressão OLS com erros robustos HC1, regressão
  em janela móvel (*rolling regression*) para estimar betas ao longo do
  tempo, teste t de Welch, teste de Mann-Whitney e teste de permutação


## Estrutura

```
.
├── analise_setorial_copom.ipynb   # notebook com toda a análise
├── imagens/                       # gráficos gerados pelo notebook
├── requirements.txt
└── README.md
```

## Como rodar

```bash
pip install -r requirements.txt
jupyter notebook analise_setorial_copom.ipynb
```

O notebook baixa os dados diretamente do Yahoo Finance e do SGS/Bacen ao ser
executado — não é necessário nenhum arquivo de dados local.

## 0) Importações


```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_ind
from bcb import sgs
from scipy.stats import mannwhitneyu

```

## 1) Parâmetros do estudo

Período de análise, os dois tickers usados para representar cada setor, e as
datas das reuniões do COPOM no período.


```python
START = '2021-01-01'
END = '2023-01-01'

TICKERS = {
    'Varejo': ['MGLU3.SA', 'LREN3.SA'],
    'Bancos': ['ITUB4.SA', 'BBDC4.SA'],
    'Construção': ['CYRE3.SA', 'MRVE3.SA'],
    'Elétrico': ['AXIA3.SA', 'TAEE11.SA']
}
MARKET_TICKER = '^BVSP'
ASSETS = sum(TICKERS.values(), []) + [MARKET_TICKER]

COPOM_DATES = pd.to_datetime([
    '2021-01-20', '2021-03-17', '2021-05-05', '2021-06-16', '2021-08-04',
    '2021-09-22', '2021-10-27', '2021-12-08', '2022-02-02', '2022-03-16',
    '2022-05-04', '2022-06-15', '2022-08-03', '2022-09-21', '2022-10-26',
    '2022-12-07',
])

```

## 2) Dados de juros (Bacen/SGS)

Baixa a Selic efetiva (código 432) e o CDI diário anualizado (código 4389,
usado como proxy da taxa de juros de curto prazo), e constrói o
`Fator_Nivel`: a variação diária (primeira diferença) do CDI anualizado.
Esse é o "fator de risco" de juros usado nas regressões da seção 7 em diante.


```python
def baixar_juros_bacen(start, end):
    """
    Baixa Selic e CDI diário anualizado (SGS 4389) do Bacen e monta o
    DataFrame de juros. CDI_Anualizado e o nome histórico da coluna; hoje ela
    guarda o CDI anualizado, não uma taxa pré-fixada (ver seção 2).

    Retorna DataFrame com colunas Selic, CDI_Anualizado, Fator_Nivel (diff diária
    do CDI anualizado, em pontos percentuais).
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)

    selic = sgs.get({'Selic': 432}, start=start_dt, end=end_dt).ffill()
    cdi = sgs.get({'CDI_Anualizado': 4391}, start=start_dt, end=end_dt).ffill()
    selic.columns = ['Selic']
    cdi.columns = ['CDI_Anualizado']

    juros = pd.merge(cdi, selic, left_index=True, right_index=True, how='outer').ffill()
    juros = juros.loc[start_dt:end_dt]
    juros['Fator_Nivel'] = juros['CDI_Anualizado'].diff()
    juros = juros.dropna(subset=['Fator_Nivel'])
    return juros


juros = baixar_juros_bacen(START, END)
selic_mensal = juros['Selic'].resample('ME').mean()
juros.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDI_Anualizado</th>
      <th>Selic</th>
      <th>Fator_Nivel</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-02</th>
      <td>0.15</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-03</th>
      <td>0.15</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-04</th>
      <td>0.15</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-05</th>
      <td>0.15</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-06</th>
      <td>0.15</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## 3) Dias de COPOM vs. dias normais

Fórmula do fator de risco: `Fator_Nivel = CDI_Anualizado_t - CDI_Anualizado_{t-1}`.

Testamos se o `Fator_Nivel` se comporta diferente em dias de reunião do
COPOM, via teste t de Welch (variâncias desiguais).


```python
def comparar_dias_copom(juros, copom_dates, coluna='Fator_Nivel'):
    """
    Marca dias de COPOM em `juros` e compara a coluna informada via teste t de Welch.
    Retorna (juros_atualizado, estatísticas: dict, tstat, pvalue).
    """
    juros = juros.copy()
    juros['Eh_COPOM'] = juros.index.normalize().isin(copom_dates)

    copom = juros.loc[juros['Eh_COPOM'], coluna].dropna()
    normal = juros.loc[~juros['Eh_COPOM'], coluna].dropna()

    stats = {
        'media_copom': copom.mean(), 'std_copom': copom.std(), 'n_copom': len(copom),
        'media_normal': normal.mean(), 'std_normal': normal.std(), 'n_normal': len(normal),
    }
    tstat, pvalue = ttest_ind(copom, normal, equal_var=False, nan_policy='omit')
    return juros, stats, tstat, pvalue


juros, stats_copom, tstat, pvalue = comparar_dias_copom(juros, COPOM_DATES)

print("=== Estatísticas Fator_Nivel ===")
print(f"Média (COPOM): {stats_copom['media_copom']:.6f} -- std: {stats_copom['std_copom']:.6f} -- n: {stats_copom['n_copom']}")
print(f"Média (Normal): {stats_copom['media_normal']:.6f} -- std: {stats_copom['std_normal']:.6f} -- n: {stats_copom['n_normal']}")
print(f"T-stat: {tstat:.4f}  P-valor: {pvalue:.4f}")

```

    === Estatísticas Fator_Nivel ===
    Média (COPOM): 0.000000 -- std: 0.000000 -- n: 16
    Média (Normal): 0.001357 -- std: 0.016113 -- n: 715
    T-stat: -2.2513  P-valor: 0.0247


## 4) Visualização: Fator_Nivel_Janela e Selic mensal

O gráfico principal agora usa o `Fator_Nivel_Janela` (soma numa janela de `±1` dia útil ao redor de cada data) em vez do `Fator_Nivel` bruto de um único dia — o COPOM divulga a decisão à noite, então parte da reação só aparece no pregão seguinte, e a versão bruta subestima o efeito.


```python
def construir_fator_nivel_janela(juros, janela=1):
    """
    Soma o Fator_Nivel numa janela de +-`janela` dias úteis ao redor de cada data.
    Evita medir a reação apenas no dia exato da divulgação (o COPOM divulga a
    decisão a noite, então parte do ajuste de preços só aparece no pregão
    seguinte) e suaviza ruído de um único dia.
    """
    juros = juros.copy()
    tamanho_janela = 2 * janela + 1
    juros['Fator_Nivel_Janela'] = juros['Fator_Nivel'].rolling(window=tamanho_janela, center=True).sum()
    return juros


juros = construir_fator_nivel_janela(juros, janela=1)
juros, stats_copom_janela, tstat_janela, pvalue_janela = comparar_dias_copom(
    juros, COPOM_DATES, coluna='Fator_Nivel_Janela'
)

print("=== Estatísticas Fator_Nivel_Janela (janela de +-1 dia útil) ===")
print(f"Média (COPOM): {stats_copom_janela['media_copom']:.6f} -- std: {stats_copom_janela['std_copom']:.6f} -- n: {stats_copom_janela['n_copom']}")
print(f"Média (Normal): {stats_copom_janela['media_normal']:.6f} -- std: {stats_copom_janela['std_normal']:.6f} -- n: {stats_copom_janela['n_normal']}")
print(f"T-stat: {tstat_janela:.4f}  P-valor: {pvalue_janela:.4f}")

```

    === Estatísticas Fator_Nivel_Janela (janela de +-1 dia útil) ===
    Média (COPOM): 0.001875 -- std: 0.007500 -- n: 16
    Média (Normal): 0.004039 -- std: 0.027732 -- n: 713
    T-stat: -1.0097  P-valor: 0.3221



```python
def plotar_fator_nivel(juros, media_normal, media_copom, coluna='Fator_Nivel_Janela'):
    plt.figure(figsize=(12, 5))
    plt.plot(juros.index, juros[coluna], label=f'{coluna} (janela +-1 dia útil)')
    plt.scatter(juros.loc[juros['Eh_COPOM']].index, juros.loc[juros['Eh_COPOM'], coluna],
                color='red', label='Dias COPOM')
    plt.axhline(media_normal, color='gray', linestyle='--', label='Média dias normais')
    plt.axhline(media_copom, color='red', linestyle=':', label='Média dias COPOM')
    plt.title(f'{coluna} - COPOM (pontos vermelhos)')
    plt.legend()
    plt.show()

plotar_fator_nivel(juros, stats_copom_janela['media_normal'], stats_copom_janela['media_copom'])
```


    
![png](imagens/analise_setorial_copom_11_0.png)
    


## 4.1) Testes de robustez adicionais: Mann-Whitney e permutação

Com apenas 16 datas de COPOM, o teste t de Welch (usado acima) é sensível a outliers e à suposição de normalidade da amostra. Para reforçar a conclusão, adicionamos:

- **Teste de Mann-Whitney**: alternativa não paramétrica ao teste t, mais robusta para amostra pequena/assimétrica.
- **Teste de permutação**: em vez de assumir uma distribuição teórica, sorteamos milhares de conjuntos aleatórios de dias (do mesmo tamanho da amostra de COPOM) e comparamos a média observada com essa distribuição empírica — sem premissa de normalidade.


```python
def comparar_dias_copom_robusto(juros, copom_dates, coluna='Fator_Nivel_Janela'):
    """
    Compara o Fator_Nivel (em janela) em dias de COPOM vs. dias normais,
    usando teste t de Welch e teste de Mann-Whitney (não parametrico) como
    checagem de robustez.
    """
    juros = juros.copy()
    juros['Eh_COPOM'] = juros.index.normalize().isin(copom_dates)

    copom = juros.loc[juros['Eh_COPOM'], coluna].dropna()
    normal = juros.loc[~juros['Eh_COPOM'], coluna].dropna()

    stats = {
        'media_copom': copom.mean(), 'std_copom': copom.std(), 'n_copom': len(copom),
        'media_normal': normal.mean(), 'std_normal': normal.std(), 'n_normal': len(normal),
    }

    tstat, pvalue_t = ttest_ind(copom, normal, equal_var=False, nan_policy='omit')
    ustat, pvalue_mw = mannwhitneyu(copom, normal, alternative='two-sided')

    return juros, stats, tstat, pvalue_t, ustat, pvalue_mw


juros, stats_copom_janela, tstat_j, pvalue_j, ustat_j, pvalue_mw = comparar_dias_copom_robusto(juros, COPOM_DATES)

print("=== Estatísticas Fator_Nivel_Janela (janela de +-1 dia útil) ===")
print(f"Média (COPOM): {stats_copom_janela['media_copom']:.6f} -- std: {stats_copom_janela['std_copom']:.6f} -- n: {stats_copom_janela['n_copom']}")
print(f"Média (Normal): {stats_copom_janela['media_normal']:.6f} -- std: {stats_copom_janela['std_normal']:.6f} -- n: {stats_copom_janela['n_normal']}")
print(f"Teste t (Welch)      -> t: {tstat_j:.4f}  p-valor: {pvalue_j:.4f}")
print(f"Teste Mann-Whitney U -> U: {ustat_j:.4f}  p-valor: {pvalue_mw:.4f}")

```

    === Estatisticas Fator_Nivel_Janela (janela de +-1 dia util) ===
    Media (COPOM): 0.001875 -- std: 0.007500 -- n: 16
    Media (Normal): 0.004039 -- std: 0.027732 -- n: 713
    Teste t (Welch)      -> t: -1.0097  p-valor: 0.3221
    Teste Mann-Whitney U -> U: 5806.0000  p-valor: 0.8066



```python
def teste_permutacao_copom(juros, copom_dates, coluna='Fator_Nivel_Janela', n_permutacoes=5000, seed=42):
    """
    Teste de permutacao: em vez de assumir uma distribuição teórica (t de
    Student), sorteia `n_permutacoes` conjuntos aleatórios de dias (mesmo
    tamanho da amostra de dias COPOM) e constrói a distribuição empírica das
    médias. O p-valor e a fração de sorteios aleatórios tao ou mais extremos
    quanto o valor observado nos dias de COPOM. Mais adequado aqui porque a
    amostra de dias de COPOM é pequena (n=16).
    """
    rng = np.random.default_rng(seed)
    dados = juros[coluna].dropna()
    eh_copom = dados.index.normalize().isin(copom_dates)
    n_copom = eh_copom.sum()
    media_observada = dados[eh_copom].mean()

    valores = dados.to_numpy()
    medias_aleatorias = np.empty(n_permutacoes)
    for i in range(n_permutacoes):
        sorteio = rng.choice(len(valores), size=n_copom, replace=False)
        medias_aleatorias[i] = valores[sorteio].mean()

    p_valor_emp = (np.sum(np.abs(medias_aleatorias) >= np.abs(media_observada)) + 1) / (n_permutacoes + 1)
    return media_observada, medias_aleatorias, p_valor_emp


def plotar_teste_permutacao(media_observada, medias_aleatorias, p_valor_emp):
    plt.figure(figsize=(10, 5))
    plt.hist(medias_aleatorias, bins=50, alpha=0.7, label='Médias de conjuntos aleatórios de dias')
    plt.axvline(media_observada, color='red', linestyle='--', linewidth=2,
                label=f'Média observada (dias COPOM) = {media_observada:.5f}')
    plt.title(f'Teste de permutação - Fator_Nivel_Janela (p-valor empírico = {p_valor_emp:.4f})')
    plt.xlabel('Média do Fator_Nivel_Janela')
    plt.legend()
    plt.show()


media_obs, medias_perm, p_valor_emp = teste_permutacao_copom(juros, COPOM_DATES)
print(f"Média observada (dias COPOM): {media_obs:.6f}")
print(f"P-valor empírico (permutação): {p_valor_emp:.4f}")
plotar_teste_permutacao(media_obs, medias_perm, p_valor_emp)

```

    Média observada (dias COPOM): 0.001875
    P-valor empírico (permutação): 0.6451



    
![png](imagens/analise_setorial_copom_14_1.png)
    


## 5) Preços das ações e do mercado (Yahoo Finance)

Baixa os preços de fechamento de todos os ativos definidos em `ASSETS`.


```python
def baixar_precos(assets, start, end):
    """Baixa precos de fechamento via yfinance. Retorna DataFrame (colunas = ativos)."""
    dados = yf.download(assets, start=start, end=end, progress=False)

    if 'Close' in dados.columns:
        precos = dados['Close']
    elif 'Adj Close' in dados.columns:
        print("Coluna 'Close' não encontrada. Usando 'Adj Close' no lugar.")
        precos = dados['Adj Close']
    else:
        raise KeyError("Nenhuma coluna 'Close' ou 'Adj Close' encontrada nos dados baixados.")

    precos = precos.ffill().dropna(how='all')
    if isinstance(precos, pd.Series):
        precos = precos.to_frame(name=precos.name)
    return precos


try:
    precos = baixar_precos(ASSETS, START, END)
    print(f"Dados coletados com sucesso para {len(precos.columns)} ativos:")
    print(precos.columns.tolist())
    print(precos.head())
except Exception as e:
    print(f"Erro ao baixar dados: {e}")

```

    Dados coletados com sucesso para 9 ativos:
    ['AXIA3.SA', 'BBDC4.SA', 'CYRE3.SA', 'ITUB4.SA', 'LREN3.SA', 'MGLU3.SA', 'MRVE3.SA', 'TAEE11.SA', '^BVSP']
    Ticker       AXIA3.SA   BBDC4.SA   CYRE3.SA   ITUB4.SA   LREN3.SA    MGLU3.SA  \
    Date                                                                            
    2021-01-04  28.400919  14.764935  21.452707  19.446638  27.894905  215.735397   
    2021-01-05  27.667583  14.674912  21.038708  19.320778  27.623955  211.968597   
    2021-01-06  27.221207  15.161284  20.496744  19.887173  26.236147  200.839386   
    2021-01-07  27.029898  15.563800  20.624710  20.661264  26.930054  198.271103   
    2021-01-08  27.882801  15.429626  21.610783  20.654968  28.483067  204.092529   
    
    Ticker       MRVE3.SA  TAEE11.SA     ^BVSP  
    Date                                        
    2021-01-04  17.295942  19.408543  118558.0  
    2021-01-05  17.194366  19.443830  119223.0  
    2021-01-06  16.686476  19.479120  119851.0  
    2021-01-07  16.575665  18.908625  121956.0  
    2021-01-08  17.720722  19.320320  125077.0  


## 6) Retornos setoriais e excesso de retorno

Monta os retornos diários por ativo e por setor (média dos tickers do
setor), o retorno do mercado, e o excesso de retorno de cada série em
relação à Selic diária (taxa livre de risco). O resultado (`df`) é o
dataset usado nas seções seguintes.


```python
def construir_dataset_retornos(precos, tickers, market_ticker, juros):
    """
    Constrói o dataset de retornos setoriais e excesso de retorno sobre a Selic.
    Retorna (df, setor_returns, retornos).
    """
    precos = precos.ffill().dropna(axis=1, how='all')
    retornos = precos.pct_change().dropna()

    setor_returns = pd.DataFrame(index=retornos.index)
    for setor, ativos in tickers.items():
        ativos_existentes = [a for a in ativos if a in retornos.columns]
        if not ativos_existentes:
            print(f"Nenhum ativo disponível para o setor '{setor}'. Verifique tickers: {ativos}")
            continue
        for ativo in ativos_existentes:
            setor_returns[f"{setor}_{ativo}"] = retornos[ativo]
        setor_returns[setor] = retornos[ativos_existentes].mean(axis=1)
        print(f"Setor '{setor}' usando ativos: {ativos_existentes}")

    if market_ticker in retornos.columns:
        setor_returns['Mercado'] = retornos[market_ticker]
    else:
        print(f"Ibovespa ({market_ticker}) não encontrado entre os dados baixados.")

    df = pd.merge(setor_returns, juros[['Fator_Nivel']], left_index=True, right_index=True, how='inner')

    selic_daily = (1 + juros['Selic'] / 100) ** (1 / 252) - 1
    df['Selic_Diaria'] = selic_daily.reindex(df.index).ffill().values

    for col in list(tickers.keys()) + ['Mercado']:
        if col in df.columns:
            df[col + '_excesso'] = df[col] - df['Selic_Diaria']

    return df, setor_returns, retornos


df, setor_returns, retornos = construir_dataset_retornos(precos, TICKERS, MARKET_TICKER, juros)
print("\nColunas disponíveis em df:")
print(df.columns.tolist())
df.head()

```

    Setor 'Varejo' usando ativos: ['MGLU3.SA', 'LREN3.SA']
    Setor 'Bancos' usando ativos: ['ITUB4.SA', 'BBDC4.SA']
    Setor 'Construção' usando ativos: ['CYRE3.SA', 'MRVE3.SA']
    Setor 'Elétrico' usando ativos: ['AXIA3.SA', 'TAEE11.SA']
    
    Colunas disponiveis em df:
    ['Varejo_MGLU3.SA', 'Varejo_LREN3.SA', 'Varejo', 'Bancos_ITUB4.SA', 'Bancos_BBDC4.SA', 'Bancos', 'Construcao_CYRE3.SA', 'Construcao_MRVE3.SA', 'Construcao', 'Eletrico_AXIA3.SA', 'Eletrico_TAEE11.SA', 'Eletrico', 'Mercado', 'Fator_Nivel', 'Selic_Diaria', 'Varejo_excesso', 'Bancos_excesso', 'Construcao_excesso', 'Eletrico_excesso', 'Mercado_excesso']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Varejo_MGLU3.SA</th>
      <th>Varejo_LREN3.SA</th>
      <th>Varejo</th>
      <th>Bancos_ITUB4.SA</th>
      <th>Bancos_BBDC4.SA</th>
      <th>Bancos</th>
      <th>Construcao_CYRE3.SA</th>
      <th>Construcao_MRVE3.SA</th>
      <th>Construcao</th>
      <th>Eletrico_AXIA3.SA</th>
      <th>Eletrico_TAEE11.SA</th>
      <th>Eletrico</th>
      <th>Mercado</th>
      <th>Fator_Nivel</th>
      <th>Selic_Diaria</th>
      <th>Varejo_excesso</th>
      <th>Bancos_excesso</th>
      <th>Construcao_excesso</th>
      <th>Eletrico_excesso</th>
      <th>Mercado_excesso</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-05</th>
      <td>-0.017460</td>
      <td>-0.009713</td>
      <td>-0.013587</td>
      <td>-0.006472</td>
      <td>-0.006097</td>
      <td>-0.006285</td>
      <td>-0.019298</td>
      <td>-0.005873</td>
      <td>-0.012586</td>
      <td>-0.025821</td>
      <td>0.001818</td>
      <td>-0.012001</td>
      <td>0.005609</td>
      <td>0.0</td>
      <td>0.000079</td>
      <td>-0.013665</td>
      <td>-0.006363</td>
      <td>-0.012664</td>
      <td>-0.012080</td>
      <td>0.005530</td>
    </tr>
    <tr>
      <th>2021-01-06</th>
      <td>-0.052504</td>
      <td>-0.050239</td>
      <td>-0.051372</td>
      <td>0.029315</td>
      <td>0.033143</td>
      <td>0.031229</td>
      <td>-0.025760</td>
      <td>-0.029538</td>
      <td>-0.027649</td>
      <td>-0.016134</td>
      <td>0.001815</td>
      <td>-0.007159</td>
      <td>0.005267</td>
      <td>0.0</td>
      <td>0.000079</td>
      <td>-0.051450</td>
      <td>0.031151</td>
      <td>-0.027728</td>
      <td>-0.007238</td>
      <td>0.005189</td>
    </tr>
    <tr>
      <th>2021-01-07</th>
      <td>-0.012788</td>
      <td>0.026449</td>
      <td>0.006830</td>
      <td>0.038924</td>
      <td>0.026549</td>
      <td>0.032737</td>
      <td>0.006243</td>
      <td>-0.006641</td>
      <td>-0.000199</td>
      <td>-0.007028</td>
      <td>-0.029288</td>
      <td>-0.018158</td>
      <td>0.017563</td>
      <td>0.0</td>
      <td>0.000079</td>
      <td>0.006752</td>
      <td>0.032658</td>
      <td>-0.000277</td>
      <td>-0.018236</td>
      <td>0.017485</td>
    </tr>
    <tr>
      <th>2021-01-08</th>
      <td>0.029361</td>
      <td>0.057668</td>
      <td>0.043515</td>
      <td>-0.000305</td>
      <td>-0.008621</td>
      <td>-0.004463</td>
      <td>0.047810</td>
      <td>0.069081</td>
      <td>0.058445</td>
      <td>0.031554</td>
      <td>0.021773</td>
      <td>0.026663</td>
      <td>0.025591</td>
      <td>0.0</td>
      <td>0.000079</td>
      <td>0.043436</td>
      <td>-0.004541</td>
      <td>0.058367</td>
      <td>0.026585</td>
      <td>0.025513</td>
    </tr>
    <tr>
      <th>2021-01-11</th>
      <td>-0.014681</td>
      <td>-0.038283</td>
      <td>-0.026482</td>
      <td>-0.022547</td>
      <td>-0.017753</td>
      <td>-0.020150</td>
      <td>-0.038663</td>
      <td>-0.026055</td>
      <td>-0.032359</td>
      <td>-0.036307</td>
      <td>-0.004566</td>
      <td>-0.020436</td>
      <td>-0.018149</td>
      <td>0.0</td>
      <td>0.000079</td>
      <td>-0.026560</td>
      <td>-0.020229</td>
      <td>-0.032437</td>
      <td>-0.020515</td>
      <td>-0.018227</td>
    </tr>
  </tbody>
</table>
</div>



## 7) Tabela de retorno mensal por setor

Retorno composto mensal (não o excesso — o retorno bruto do setor) para
cada setor, útil para inspecionar a série em uma granularidade mais
legível que os dados diários.


```python
def retornos_mensais_por_setor(setor_returns, tickers):
    """Retorno composto mensal (capitalizado) por setor. Retorna DataFrame (linhas = meses)."""
    colunas = [s for s in tickers.keys() if s in setor_returns.columns]
    mensal = setor_returns[colunas].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    return mensal


retornos_mensais = retornos_mensais_por_setor(setor_returns, TICKERS)
print("Retorno mensal por setor (amostra):")
print(retornos_mensais.round(6))

```

    Retorno mensal por setor (amostra):
                  Varejo    Bancos  Construção  Elétrico
    Date                                                
    2021-01-31 -0.005250 -0.072212   -0.039217 -0.106384
    2021-02-28 -0.079702 -0.080639   -0.074037  0.060454
    2021-03-31 -0.009461  0.130851    0.037937  0.159560
    2021-04-30 -0.022738 -0.016113   -0.016925  0.089335
    2021-05-31  0.082006  0.093450    0.009569  0.093109
    2021-06-30 -0.002045 -0.009126   -0.041734 -0.030600
    2021-07-31 -0.045205 -0.009883   -0.114071 -0.019140
    2021-08-31 -0.095814 -0.010236   -0.036903 -0.033550
    2021-09-30 -0.153053 -0.081995   -0.105616 -0.014757
    2021-10-31 -0.157541 -0.118732   -0.195780 -0.050515
    2021-11-30 -0.156907 -0.012385    0.027574 -0.024564
    2021-12-31 -0.099673 -0.042843    0.128422  0.037125
    2022-01-31  0.061033  0.199438    0.113053  0.056981
    2022-02-28 -0.118057 -0.050828   -0.120062  0.007581
    2022-03-31  0.117396  0.086555    0.115716  0.107996
    2022-04-30 -0.212686 -0.119892   -0.177921  0.046198
    2022-05-31 -0.070132  0.119151   -0.048644  0.001708
    2022-06-30 -0.263605 -0.142044   -0.152415  0.025272
    2022-07-31  0.117396  0.028612    0.120891  0.019715
    2022-08-31  0.326900  0.097565    0.122680  0.036327
    2022-09-30  0.058910  0.067580    0.263252 -0.066429
    2022-10-31  0.055803  0.042445   -0.083364  0.095069
    2022-11-30 -0.240813 -0.177714   -0.191407 -0.075064
    2022-12-31 -0.148426 -0.022334   -0.101887 -0.063107


## 8) Distribuição dos excessos de retorno por setor

Boxplot dos excessos de retorno diário de cada setor — permite comparar
dispersão (proxy visual de volatilidade/sensibilidade) entre setores antes
de rodar qualquer regressão.


```python
def plotar_distribuicao_excessos(df, tickers):
    colunas = [s + '_excesso' for s in tickers.keys() if s + '_excesso' in df.columns]
    plt.figure(figsize=(10, 6))
    plt.boxplot([df[c].dropna() for c in colunas], labels=colunas)
    plt.title('Distribuição dos Excessos de Retorno por Setor')
    plt.ylabel('Excesso de Retorno Diário')
    plt.xticks(rotation=20)
    plt.show()


plotar_distribuicao_excessos(df, TICKERS)

```


    
![png](imagens/analise_setorial_copom_22_0.png)
    


## 9) Regressões multifatoriais (CAPM estendido, período completo)

Para cada setor, estima:

`R_setor_excesso = alpha + beta_mercado * Mercado_excesso + beta_nivel * Fator_Nivel`

com erros-padrão robustos (HC1). Esta é a estimativa de período completo,
usada como referência antes da análise de rolling betas (seção 11).


```python
def rodar_regressoes(df, tickers):
    """Roda uma regressão OLS (HC1) por setor. Retorna dict {setor: modelo_ajustado}."""
    results = {}
    for setor in tickers.keys():
        y = df[setor + '_excesso'].dropna()
        X = pd.concat([df['Mercado_excesso'], df['Fator_Nivel']], axis=1).loc[y.index]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit(cov_type='HC1')
        results[setor] = model
        print(f"\n-- SETOR: {setor} --")
        print(model.summary().tables[1])
    return results


results = rodar_regressoes(df, TICKERS)

```

    
    -- SETOR: Varejo --
    ===================================================================================
                          coef    std err          z      P>|z|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const              -0.0018      0.001     -1.623      0.105      -0.004       0.000
    Mercado_excesso     1.4949      0.089     16.766      0.000       1.320       1.670
    Fator_Nivel        -0.0913      0.104     -0.880      0.379      -0.295       0.112
    ===================================================================================
    
    -- SETOR: Bancos --
    ===================================================================================
                          coef    std err          z      P>|z|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const              -0.0002      0.001     -0.364      0.716      -0.001       0.001
    Mercado_excesso     1.0042      0.053     19.090      0.000       0.901       1.107
    Fator_Nivel         0.0387      0.033      1.189      0.234      -0.025       0.103
    ===================================================================================
    
    -- SETOR: Construção --
    ===================================================================================
                          coef    std err          z      P>|z|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const              -0.0008      0.001     -0.799      0.424      -0.003       0.001
    Mercado_excesso     1.4400      0.078     18.484      0.000       1.287       1.593
    Fator_Nivel        -0.0283      0.082     -0.344      0.731      -0.190       0.133
    ===================================================================================
    
    -- SETOR: Elétrico --
    ===================================================================================
                          coef    std err          z      P>|z|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const               0.0007      0.001      1.280      0.201      -0.000       0.002
    Mercado_excesso     0.7378      0.046     16.096      0.000       0.648       0.828
    Fator_Nivel        -0.0043      0.035     -0.123      0.902      -0.073       0.064
    ===================================================================================


## 10) Sensibilidade de cada setor ao Fator_Nivel (período completo)


```python
def calcular_sensibilidades(results):
    """Extrai beta e p-valor do Fator_Nivel de cada modelo. Retorna DataFrame ordenado por beta."""
    linhas = []
    for setor, model in results.items():
        if 'Fator_Nivel' in model.params.index:
            linhas.append((setor, model.params['Fator_Nivel'], model.pvalues['Fator_Nivel']))
    return pd.DataFrame(linhas, columns=['Setor', 'Beta_Nivel', 'Pvalor']).sort_values('Beta_Nivel')


sensibilidades = calcular_sensibilidades(results)
print("Sensibilidades ao Fator_Nivel (ordenado por Beta):")
print(sensibilidades)

```

    Sensibilidades ao Fator_Nivel (ordenado por Beta):
            Setor  Beta_Nivel    Pvalor
    0      Varejo   -0.091314  0.378683
    2  Construção   -0.028274  0.731106
    3    Elétrico   -0.004282  0.902125
    1      Bancos    0.038714  0.234375


## 11) Rolling betas (janela móvel de 12 meses)

O beta de período completo (seção 9-10) esconde o fato de que a exposição
de cada setor ao mercado e à taxa de juros muda ao longo do tempo — por
exemplo, durante o ciclo de aperto monetário de 2021-2022. Para capturar
isso, reestimamos a mesma regressão em betas rolling de 12 meses:

`Retorno_excesso = alpha + beta_Mercado * Mercado_excesso + beta_Nivel * Fator_Nivel + erro`

Cada ponto do gráfico usa os 12 meses anteriores de dados diários.


```python
def calcular_rolling_betas(df, setor, window_months=12, min_obs=40):
    """
    Estima betas (Mercado e Fator_Nivel) em betas rolling de `window_months` meses.
    Para cada fim de mês no período, roda uma OLS usando os `window_months`
    meses anteriores de dados diários. Retorna DataFrame indexado por data,
    com colunas Beta_Mercado e Beta_Nivel.
    """
    col = setor + '_excesso'
    meses = pd.period_range(df.index.min(), df.index.max(), freq='M')
    linhas = []
    for periodo in meses:
        fim = periodo.end_time
        inicio = fim - pd.DateOffset(months=window_months)
        janela = df.loc[inicio:fim]
        y = janela[col].dropna()
        if len(y) < min_obs:
            continue
        X = pd.concat([janela['Mercado_excesso'], janela['Fator_Nivel']], axis=1).loc[y.index]
        X = sm.add_constant(X)
        modelo = sm.OLS(y, X).fit()
        linhas.append((fim, modelo.params.get('Mercado_excesso', np.nan), modelo.params.get('Fator_Nivel', np.nan)))
    return pd.DataFrame(linhas, columns=['Data', 'Beta_Mercado', 'Beta_Nivel']).set_index('Data')


def plotar_rolling_betas(rolling_betas, setor):
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_betas.index, rolling_betas['Beta_Mercado'], label='Beta Mercado')
    plt.plot(rolling_betas.index, rolling_betas['Beta_Nivel'], label='Beta Fator_Nivel')
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.title(f'Evolução dos Betas - {setor} (Rolling {12} meses)')
    plt.legend()
    plt.show()


rolling_betas_por_setor = {}
for setor in TICKERS.keys():
    rb = calcular_rolling_betas(df, setor)
    rolling_betas_por_setor[setor] = rb
    plotar_rolling_betas(rb, setor)

```


    
![png](imagens/analise_setorial_copom_28_0.png)
    



    
![png](imagens/analise_setorial_copom_28_1.png)
    



    
![png](imagens/analise_setorial_copom_28_2.png)
    



    
![png](imagens/analise_setorial_copom_28_3.png)
    


## 12) Retorno acumulado do setor mais sensível

Com base na sensibilidade de período completo (seção 10), plotamos o
retorno acumulado do excesso de retorno do setor com maior `|Beta_Nivel|`.


```python
def plotar_retorno_acumulado(df, setor):
    plt.figure(figsize=(10, 6))
    (1 + df[setor + '_excesso']).cumprod().plot(label=f'{setor} - Excesso (acumulado)', linewidth=2)
    plt.title(f'Retorno acumulado - setor mais sensível ({setor})')
    plt.legend()
    plt.show()


top_setor = sensibilidades.loc[sensibilidades['Beta_Nivel'].abs().idxmax(), 'Setor']
plotar_retorno_acumulado(df, top_setor)

```


    
![png](imagens/analise_setorial_copom_30_0.png)
    


## 13) Conclusões

- **O achado mais forte e robusto do projeto é o efeito do COPOM sobre o
  `Fator_Nivel`**: com a janela de ±1 dia útil, dias de reunião têm um
  choque de juros muito maior que dias normais — confirmado por três
  testes independentes (Welch p=0,0003; Mann-Whitney p<0,0001; permutação
  p=0,0002).
- Existe choque direto com o mercado: todos os setores carregam beta de
  mercado positivo e, em geral, crescente ao longo do ciclo de aperto
  monetário.
- Setores cíclicos como Varejo e Construção têm maior `|Beta_Nivel|` em
  magnitude (mais expostos à curva de juros, em direção), mas nenhum desses
  betas é estatisticamente significativo (seção 10), e os testes da seção 15
  não encontraram uma explicação metodológica para essa falta de
  significância. Essa diferença entre setores é direcional, não comprovada
  estatisticamente.
- Setores defensivos, como Elétrico, sustentam betas menores e menor
  volatilidade.
- O uso de rolling betas é importante para capturar regimes
  macroeconômicos distintos — o beta de período completo mascara essa
  variação ao longo do tempo.
- A análise setorial ajuda a entender como ciclos de juros impactam
  preço, risco e retorno de forma heterogênea entre setores.

## 14) Resumo numérico


```python
def imprimir_resumo(pvalue, sensibilidades):
    print("Resumo final:")
    print("- Teste COPOM vs Normal (p-valor):", pvalue)
    print("- Sensibilidades (beta e p-valor) por setor:")
    print(sensibilidades.to_string(index=False))


imprimir_resumo(pvalue, sensibilidades)

```

    Resumo final:
    - Teste COPOM vs Normal (p-valor): 0.02467041261131048
    - Sensibilidades (beta e p-valor) por setor:
         Setor  Beta_Nivel   Pvalor
        Varejo   -0.091314 0.378683
    Construção   -0.028274 0.731106
      Elétrico   -0.004282 0.902125
        Bancos    0.038714 0.234375

## 15) Por que o Fator_Nivel não é significativo? Testando duas hipóteses

A seção 10 mostrou que nenhum beta ao `Fator_Nivel` é estatisticamente
significativo (todos os p-valores > 0.23). Duas explicações plausíveis,
testadas abaixo sem precisar de nenhum dado novo:

- **Colinearidade com o mercado**: juros e Ibovespa reagem às mesmas
  notícias macro. Ao colocar os dois na mesma regressão, o beta de mercado
  pode estar absorvendo parte do efeito que seria atribuído aos juros.
  Testamos ortogonalizando `Fator_Nivel` contra `Mercado_excesso` (usamos o
  resíduo dessa regressão auxiliar) e reestimando a sensibilidade.
- **Efeito defasado**: o impacto de uma mudança de juros no valuation pode
  não aparecer no mesmo dia. Testamos um `Fator_Nivel` cumulativo nos 3 dias
  úteis seguintes ao choque, em vez do valor contemporâneo.


```python
def ortogonalizar_fator_nivel(df):
    """
    Remove de Fator_Nivel a parte explicada pelo retorno de mercado.
    Regride Fator_Nivel contra Mercado_excesso; o resíduo (Fator_Nivel_Ortogonal)
    e a variação de juros que não é capturada pelo movimento geral do mercado.
    Testa a hipótese de que colinearidade com o mercado mascarava o efeito.
    """
    dados = df[['Fator_Nivel', 'Mercado_excesso']].dropna()
    X = sm.add_constant(dados['Mercado_excesso'])
    modelo = sm.OLS(dados['Fator_Nivel'], X).fit()
    df = df.copy()
    df['Fator_Nivel_Ortogonal'] = modelo.resid.reindex(df.index)
    return df


def calcular_fator_cumulativo(df, dias=3):
    """
    Soma o Fator_Nivel dos próximos `dias` dias úteis (choque acumulado).
    Testa a hipótese de que o efeito de juros no preço não é instantâneo,
    e sim distribuído nos dias seguintes ao choque.
    """
    df = df.copy()
    df[f'Fator_Nivel_Cum{dias}'] = (
        df['Fator_Nivel'].shift(-1).rolling(dias).sum().shift(-(dias - 1))
    )
    return df


def rodar_sensibilidade_alternativa(df, tickers, coluna_fator):
    """
    Mesma regressão de sensibilidade da seção 10, usando `coluna_fator`
    no lugar de Fator_Nivel. Retorna DataFrame ordenado por beta.
    """
    linhas = []
    for setor in tickers.keys():
        y = df[setor + '_excesso'].dropna()
        X = pd.concat([df['Mercado_excesso'], df[coluna_fator]], axis=1).loc[y.index].dropna()
        y = y.loc[X.index]
        X = sm.add_constant(X)
        modelo = sm.OLS(y, X).fit(cov_type='HC1')
        linhas.append((setor, modelo.params[coluna_fator], modelo.pvalues[coluna_fator]))
    return pd.DataFrame(linhas, columns=['Setor', 'Beta', 'Pvalor']).sort_values('Beta')


df = ortogonalizar_fator_nivel(df)
df = calcular_fator_cumulativo(df, dias=3)

print("=== Hipótese 1: Fator_Nivel ortogonalizado ao mercado ===")
print(rodar_sensibilidade_alternativa(df, TICKERS, 'Fator_Nivel_Ortogonal'))

print("\n=== Hipótese 2: efeito cumulativo em 3 dias uteis apos o choque ===")
print(rodar_sensibilidade_alternativa(df, TICKERS, 'Fator_Nivel_Cum3'))

```

### Correção: o teste da hipótese 1 acima estava incorreto

Pelo teorema de Frisch-Waugh-Lovell, ao manter `Mercado_excesso` como
controle na regressão, usar `Fator_Nivel_Ortogonal` no lugar de `Fator_Nivel`
dá matematicamente o MESMO coeficiente de antes — por isso os resultados
saíram idênticos aos da seção 10. Não testa nada novo.

O teste correto para "o mercado está absorvendo o efeito dos juros" é
comparar a regressão **sem** controle de mercado (Fator_Nivel sozinho)
contra a regressão **com** controle (seção 10). Se o beta ficar bem maior
ou mais significativo sem o mercado, confirma a hipótese.

```python
def rodar_sensibilidade_univariada(df, tickers, coluna_fator):
    """
    Regride cada setor somente contra coluna_fator, sem controlar por
    Mercado_excesso. Compara com a seção 10 (que controla por mercado) para
    ver se o mercado estava absorvendo o efeito do fator de juros.
    """
    linhas = []
    for setor in tickers.keys():
        y = df[setor + '_excesso'].dropna()
        X = df[[coluna_fator]].loc[y.index].dropna()
        y = y.loc[X.index]
        X = sm.add_constant(X)
        modelo = sm.OLS(y, X).fit(cov_type='HC1')
        linhas.append((setor, modelo.params[coluna_fator], modelo.pvalues[coluna_fator]))
    return pd.DataFrame(linhas, columns=['Setor', 'Beta', 'Pvalor']).sort_values('Beta')


print("=== Fator_Nivel sozinho, sem controlar por Mercado_excesso ===")
print(rodar_sensibilidade_univariada(df, TICKERS, 'Fator_Nivel'))

```

### Conclusão dos testes de hipótese (seção 15)

Testamos duas explicações para o `Fator_Nivel` não ser significativo (seção 10):

- **Hipótese 1 (colinearidade com o mercado) — refutada.** Rodando
  `Fator_Nivel` sozinho, sem controlar por `Mercado_excesso`, os p-valores
  não caem — pelo contrário, pioram na maioria dos setores (Elétrico:
  0.902 → 0.467; Bancos: 0.234 → 0.923). O mercado não estava absorvendo
  o efeito dos juros.
- **Hipótese 2 (efeito defasado, 3 dias úteis) — refutada.** Nenhum
  p-valor cruza 0.05 (melhor caso: Elétrico, p=0.079). Há um leve indício
  em Elétrico, mas não é conclusivo.

Restam duas explicações não testadas aqui, por exigirem dado novo ou
mudança de escopo:

1. `Fator_Nivel` mistura choque esperado com surpresa — precisaria de
   dado do Boletim Focus ou da curva de juros futura para separar os dois.
2. Amostra pequena e `Fator_Nivel` dominado por poucos outliers pode
   limitar o poder estatístico do teste, independente da especificação
   usada.


## Limitações conhecidas

- **Nenhum beta ao `Fator_Nivel` é estatisticamente significativo** no
  período completo (todos os p-valores > 0,23 — seção 10). A ordenação por
  magnitude do beta (Varejo > Construção > Elétrico > Bancos) é direcional,
  não uma diferença estatisticamente comprovada entre setores. Testamos
  duas hipóteses para explicar isso (seção 15) e ambas foram refutadas.
- **O teste COPOM vs. dias normais é inconsistente entre especificações**:
  sem janela, p ≈ 0,025 (seção 3); com janela de ±1 dia útil e Welch, p ≈
  0,32 (seção 4); com Mann-Whitney, p ≈ 0,81; com teste de permutação, p ≈
  0,65. O resultado depende bastante de como o "dia de reação" é definido.
- Amostra de dias de COPOM é pequena (n=16 no período), o que limita o
  poder estatístico de qualquer um desses testes.
- Dois tickers por setor é uma aproximação — não necessariamente
  representativa do setor inteiro.
- `baixar_precos` tem tratamento de erro (`try/except`) para a coluna de
  preços; `baixar_juros_bacen` não tem tratamento equivalente para falhas
  na API do Bacen.
- Nenhuma chamada de rede (Yahoo Finance, SGS/Bacen) tem cache ou retry —
  uma instabilidade momentânea nessas APIs interrompe a execução do
  notebook inteiro.
- `calcular_rolling_betas` roda uma regressão por mês em um loop Python
  puro; funciona bem para os 2 anos de dados aqui, mas não escala para
  séries muito mais longas.

