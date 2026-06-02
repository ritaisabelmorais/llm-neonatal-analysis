# Este código extrai as variáveis e prevê um intervalo de peso para o dia escolhido

import os
import re
import tempfile
import requests
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "gemma-3-12b-it"

CAMPOS = {
    "Peso (g)": None,
    "Comprimento (cm)": None,
    "PC (cm)": None,
    "Fluidos totais (ml/kg/dia)": None,
    "NP (ml/kg/dia)": None,
    "NE (ml/kg/dia)": None,
    "Energia (kcal/kg/dia)": None,
    "Proteína (g/kg/dia)": None,
    "Lípidos (g/kg/dia)": None,
}

PATTERNS = {
    "Peso (g)": r"Peso:\s*([0-9]+(?:[.,][0-9]+)?)\s*g",
    "Comprimento (cm)": r"Comprimento:\s*([0-9]+(?:[.,][0-9]+)?)\s*cm",
    "PC (cm)": r"PC:\s*([0-9]+(?:[.,][0-9]+)?)\s*cm",
    "Fluidos totais (ml/kg/dia)": r"Fluidos totais:\s*([0-9]+(?:[.,][0-9]+)?)\s*ml/kg/dia",
    "NP (ml/kg/dia)": r"NP:\s*([0-9]+(?:[.,][0-9]+)?)\s*ml/kg/dia",
    "NE (ml/kg/dia)": r"NE:\s*([0-9]+(?:[.,][0-9]+)?)\s*ml/kg/dia",
    "Energia (kcal/kg/dia)": r"Energia:\s*([0-9]+(?:[.,][0-9]+)?)\s*kcal/kg/dia",
    "Proteína (g/kg/dia)": r"Prote[ií]na:\s*([0-9]+(?:[.,][0-9]+)?)\s*g/kg/dia",
    "Lípidos (g/kg/dia)": r"L[íi]pidos:\s*([0-9]+(?:[.,][0-9]+)?)\s*g/kg/dia",
}

PROMPT_TEMPLATE = """
Abaixo está um relatório clínico de um recém-nascido.
Dia identificado: D{dia}

Por favor, leia cuidadosamente o texto e extraia as seguintes informações clínicas, se estiverem disponíveis:

1. Peso (g)
2. Comprimento (cm)
3. PC (cm)
4. Fluidos totais (ml/kg/dia)
5. NP (ml/kg/dia)
6. NE (ml/kg/dia)
7. Energia (kcal/kg/dia)
8. Proteína (g/kg/dia)
9. Lípidos (g/kg/dia)

Regras para o peso:
- PN significa peso ao nascimento.
- No primeiro diário da série, o PN pode ser usado como Peso (g), porque corresponde ao peso inicial.
- Nos restantes dias, NUNCA uses PN como Peso (g).
- Usa apenas "Peso atual", "PA", "Peso hoje" ou equivalente.
- Não uses "Peso ontem" como peso atual.
- Se não existir peso atual válido, escreve "Peso: Não informado".

Responde apenas neste formato:

Peso: ___ g
Comprimento: ___ cm
PC: ___ cm
Fluidos totais: ___ ml/kg/dia
NP: ___ ml/kg/dia
NE: ___ ml/kg/dia
Energia: ___ kcal/kg/dia
Proteína: ___ g/kg/dia
Lípidos: ___ g/kg/dia

Relatório:
{texto}
"""

st.set_page_config(page_title="Extração de Dados Clínicos Neonatais", layout="wide")


def extrair_dia(nome):
    m = re.search(r"D(\d+)", nome, re.I)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)", nome)
    if m:
        return int(m.group(1))

    return 9999


def ler_txt(arquivo):
    data = arquivo.read()

    if isinstance(data, str):
        return data

    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass

    raise ValueError("Não foi possível ler o ficheiro.")


def normalizar_valor(v):
    v = str(v).strip().rstrip(".")

    if v.lower() in ["não informado", "nao informado", "", "none", "null"]:
        return None

    return v.replace(",", ".")


def extrair_campos(texto, dia=None, primeiro_dia=None):
    out = CAMPOS.copy()

    for campo, pat in PATTERNS.items():
        if campo == "Peso (g)":
            continue

        m = re.search(pat, texto, re.I)
        if m:
            out[campo] = normalizar_valor(m.group(1))

    padroes_peso_atual = [
        r"\bPA\s*([0-9]+(?:[.,][0-9]+)?)\s*g",
        r"\bPeso\s+atual:?\s*([0-9]+(?:[.,][0-9]+)?)\s*g",
        r"\bPeso\s+hoje\s*([0-9]+(?:[.,][0-9]+)?)\s*g",
        r"\bPeso\s+(?!ontem)([0-9]+(?:[.,][0-9]+)?)\s*g",
        r"\bPeso:?\s*([0-9]+(?:[.,][0-9]+)?)\s*g",
    ]

    for pat in padroes_peso_atual:
        m = re.search(pat, texto, re.I)
        if m:
            out["Peso (g)"] = normalizar_valor(m.group(1))
            return out

    if dia is not None and primeiro_dia is not None and dia == primeiro_dia:
        m = re.search(r"\bPN\s*([0-9]+(?:[.,][0-9]+)?)\s*g", texto, re.I)
        if m:
            out["Peso (g)"] = normalizar_valor(m.group(1))

    return out


def chamar_modelo(texto, dia):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "Extrai apenas a informação pedida de forma estruturada."
            },
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(texto=texto, dia=dia)
            }
        ],
        "max_tokens": 512,
        "temperature": 0.15,
    }

    r = requests.post(API_URL, json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()

    return j.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or "Sem resposta do modelo."


def chamar_llm_livre(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "Analisa séries temporais clínicas simples e responde de forma clara e estruturada."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1200,
        "temperature": 0.15,
    }

    r = requests.post(API_URL, json=payload, timeout=300)
    r.raise_for_status()
    j = r.json()

    return j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def pos_processar(df, textos, primeiro_dia):
    df = df.copy()

    for i, row in df.iterrows():
        texto = textos.get(row["Arquivo"], "")
        extraido_local = extrair_campos(texto, row["Dia"], primeiro_dia)

        for col in CAMPOS:
            if pd.isna(df.at[i, col]) or df.at[i, col] is None:
                if extraido_local[col] is not None:
                    df.at[i, col] = extraido_local[col]

    return df.where(pd.notnull(df), None)


def construir_serie_peso(df):
    df = df.sort_values(by="Dia")
    linhas = []

    for _, row in df.iterrows():
        peso = row["Peso (g)"]
        dia = row["Dia"]

        if peso is not None and not pd.isna(peso):
            linhas.append(f"Dia {dia}: {peso} g")

    return "\n".join(linhas)


def preparar_dados_peso(df):
    df_peso = df.copy()
    df_peso["Peso (g)"] = pd.to_numeric(df_peso["Peso (g)"], errors="coerce")
    df_peso = df_peso.dropna(subset=["Peso (g)"])
    df_peso = df_peso.sort_values(by="Dia")

    return df_peso[["Dia", "Peso (g)"]]


def construir_prompt_peso(serie, dias_a_prever):
    return f"""
Os valores seguintes representam a evolução do peso de um recém-nascido ao longo dos dias de vida.

O peso deve ser tratado como uma série temporal.

Pretende-se prever um intervalo provável para o peso daqui a {dias_a_prever} dia(s).

Regras importantes:
- Usa apenas os valores fornecidos.
- Não inventes valores ausentes.
- A previsão deve ser conservadora e coerente com as variações observadas.
- Dá maior importância aos valores mais recentes.
- Se a previsão for para mais de 5 dias, aumenta a largura do intervalo.
- Não uses regressão linear se o padrão sugerir descida seguida de subida.
- Deves dar um intervalo: valor mínimo provável e valor máximo provável.


Tarefas:
1. Mostra a lista dos pesos ordenados por dia.
2. Analisa a evolução do peso.
3. Identifica o padrão observado.
4. Estima o intervalo provável do peso daqui a {dias_a_prever} dia(s).
5. Indica o dia previsto.
6. Indica o grau de confiança.
7. Justifica de forma simples.
8. No fim, escreve exatamente neste formato:
Peso previsto mínimo: ___ g
Peso previsto máximo: ___ g

Série temporal:
{serie}
"""


def extrair_intervalo_peso(resposta):
    padrao_min = r"Peso previsto mínimo:\s*(\d{3,4})\s*g"
    padrao_max = r"Peso previsto máximo:\s*(\d{3,4})\s*g"

    m_min = re.search(padrao_min, resposta, re.I)
    m_max = re.search(padrao_max, resposta, re.I)

    if m_min and m_max:
        minimo = int(m_min.group(1))
        maximo = int(m_max.group(1))

        if minimo > maximo:
            minimo, maximo = maximo, minimo

        return minimo, maximo

    return None, None


def analisar_arquivos(arquivos):
    arquivos = sorted(arquivos, key=lambda a: extrair_dia(a.name))
    dias = [extrair_dia(a.name) for a in arquivos]
    primeiro_dia = min(dias)

    raw, struct, textos = [], [], {}

    pasta = tempfile.mkdtemp(prefix="extracao_clinica_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    f_raw = os.path.join(pasta, f"resultados_clinicos_{ts}.csv")
    f_struct = os.path.join(pasta, f"resultados_clinicos_{ts}_estruturado.csv")
    f_tab = os.path.join(pasta, f"resultados_clinicos_{ts}_tabela.csv")

    bar = st.progress(0)
    status = st.empty()

    for i, arq in enumerate(arquivos):
        nome = getattr(arq, "name", f"arquivo_{i + 1}.txt")

        try:
            texto = ler_txt(arq)
            textos[nome] = texto
            dia = extrair_dia(nome)
            resultado = chamar_modelo(texto, dia)

        except Exception as e:
            texto = f"Erro ao ler/processar: {e}"
            resultado = f"Erro ao processar: {e}"

        raw.append([nome, texto])
        struct.append([nome, resultado])

        status.write(f"A processar: {nome}")
        bar.progress((i + 1) / len(arquivos))

    pd.DataFrame(raw, columns=["Arquivo", "Texto Original"]).to_csv(
        f_raw, index=False, sep=";", encoding="utf-8-sig"
    )

    pd.DataFrame(struct, columns=["Arquivo", "Informações Estruturadas"]).to_csv(
        f_struct, index=False, sep=";", encoding="utf-8-sig"
    )

    df_raw = pd.DataFrame(raw, columns=["Arquivo", "Texto Original"])
    df_struct = pd.DataFrame(struct, columns=["Arquivo", "Informações Estruturadas"])

    df_final = pd.DataFrame(
        [
            extrair_campos(
                row["Informações Estruturadas"],
                extrair_dia(row["Arquivo"]),
                primeiro_dia
            )
            for _, row in df_struct.iterrows()
        ]
    )

    df_final.insert(0, "Arquivo", df_struct["Arquivo"])
    df_final.insert(1, "Dia", df_struct["Arquivo"].apply(extrair_dia))

    df_final = df_final.sort_values(by="Dia")

    df_final.to_csv(f_tab, index=False, sep=";", encoding="utf-8-sig")

    status.write("Processamento concluído.")

    return f_raw, f_struct, f_tab, df_raw, df_struct, df_final


st.title("Extração de Dados Clínicos Neonatais")

st.write(
    "Carrega ficheiros .txt, processa-os com o LM Studio, gera tabelas estruturadas "
    "e permite prever um intervalo de peso para o número de dias que escolheres."
)

uploaded_files = st.file_uploader(
    "Arrasta ou seleciona os ficheiros clínicos (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if st.button("Processar ficheiros"):
    if not uploaded_files:
        st.warning("Por favor, carrega pelo menos um ficheiro.")
    else:
        try:
            raw_file, structured_file, tabela_file, df_raw, df_struct, df_final = analisar_arquivos(uploaded_files)

            st.session_state["df_final"] = df_final
            st.session_state["tabela_file"] = tabela_file

        except Exception as e:
            st.error(f"Ocorreu um erro: {str(e)}")


if "df_final" in st.session_state:
    df_final = st.session_state["df_final"]
    tabela_file = st.session_state["tabela_file"]

    st.success("Resultados gerados com sucesso!")

    st.subheader("Tabela estruturada")
    st.dataframe(df_final, use_container_width=True)

    serie = construir_serie_peso(df_final)

    st.subheader("Série temporal do peso")
    st.text(serie)

    df_peso = preparar_dados_peso(df_final)

    st.subheader("Gráfico Peso vs Dia")

    if df_peso.empty:
        st.warning("Não existem valores de peso suficientes para construir o gráfico.")

    else:
        dias_unicos = sorted(df_peso["Dia"].unique().tolist())

        grafico_serie = alt.Chart(df_peso).mark_line(point=True, color="#1f77b4").encode(
            x=alt.X("Dia:Q",
                    title="Dia de vida",
                    axis=alt.Axis(values=dias_unicos, format="d", labelAngle=0)),
            y=alt.Y("Peso (g):Q", title="Peso (g)"),
            tooltip=["Dia", "Peso (g)"]
        ).properties(
            title="Evolução do Peso",
            width=700,
            height=400
        )
        st.altair_chart(grafico_serie, use_container_width=True)

        st.subheader("Previsão do peso por intervalo")

        dias_a_prever = st.number_input(
            "Quantos dias à frente queres prever?",
            min_value=1,
            max_value=30,
            value=1,
            step=1
        )

        if st.button("Prever intervalo de peso com LLM"):
            try:
                st.write("A enviar pedido ao LLM...")

                prompt = construir_prompt_peso(serie, dias_a_prever)
                resposta = chamar_llm_livre(prompt)

                st.subheader("Análise do LLM")
                st.write(resposta)

                previsao_min, previsao_max = extrair_intervalo_peso(resposta)

                if previsao_min is not None and previsao_max is not None:
                    ultimo_dia = int(df_peso["Dia"].max())
                    dia_previsto = ultimo_dia + int(dias_a_prever)

                    # Escala Y: margem de 50g em volta dos valores conhecidos e da previsão
                    todos_pesos = list(df_peso["Peso (g)"]) + [previsao_min, previsao_max]
                    y_min = max(0, min(todos_pesos) - 50)
                    y_max = max(todos_pesos) + 50

                    # Dados reais
                    df_real = df_peso.copy()

                    # Ponto central da previsão (média do intervalo)
                    df_prev_ponto = pd.DataFrame([{
                        "Dia": dia_previsto,
                        "Peso (g)": (previsao_min + previsao_max) / 2,
                    }])

                    # Dias a mostrar no eixo X (reais + dia previsto)
                    dias_eixo = sorted(set(df_real["Dia"].tolist() + [dia_previsto]))

                    # Linha dos dados reais (azul)
                    linha = alt.Chart(df_real).mark_line(color="#1f77b4").encode(
                        x=alt.X("Dia:Q",
                                title="Dia de vida",
                                axis=alt.Axis(values=dias_eixo, format="d", labelAngle=0)),
                        y=alt.Y("Peso (g):Q",
                                scale=alt.Scale(domain=[y_min, y_max]),
                                title="Peso (g)")
                    )

                    pontos_reais = alt.Chart(df_real).mark_circle(size=60, color="#1f77b4").encode(
                        x=alt.X("Dia:Q",
                                axis=alt.Axis(values=dias_eixo, format="d", labelAngle=0)),
                        y="Peso (g):Q",
                        tooltip=["Dia", "Peso (g)"]
                    )

                    # Barra de erro (intervalo de previsão) — laranja
                    df_intervalo_alt = pd.DataFrame([{
                        "Dia": dia_previsto,
                        "ymin": float(previsao_min),
                        "ymax": float(previsao_max)
                    }])

                    barra_erro = alt.Chart(df_intervalo_alt).mark_errorbar(
                        color="#ff7f0e", thickness=2, ticks=True
                    ).encode(
                        x=alt.X("Dia:Q",
                                axis=alt.Axis(values=dias_eixo, format="d", labelAngle=0)),
                        y=alt.Y("ymin:Q", scale=alt.Scale(domain=[y_min, y_max])),
                        y2="ymax:Q"
                    )

                    ponto_prev = alt.Chart(df_prev_ponto).mark_circle(
                        size=120, color="#ff7f0e"
                    ).encode(
                        x=alt.X("Dia:Q",
                                axis=alt.Axis(values=dias_eixo, format="d", labelAngle=0)),
                        y="Peso (g):Q",
                        tooltip=["Dia", "Peso (g)"]
                    )

                    grafico = (linha + pontos_reais + barra_erro + ponto_prev).properties(
                        title=f"Peso real (azul) + Intervalo previsto para o dia {dia_previsto} (laranja)",
                        width=700,
                        height=400
                    )

                    st.subheader("Gráfico Peso + Intervalo de Previsão do LLM")
                    st.altair_chart(grafico, use_container_width=True)
                    st.write(
                        f"Intervalo previsto: {previsao_min} g até {previsao_max} g para o dia {dia_previsto}"
                    )

                else:
                    st.warning(
                        "Não foi possível extrair automaticamente o intervalo previsto. "
                        "Confirma se o LLM escreveu exatamente: "
                        "'Peso previsto mínimo: ___ g' e 'Peso previsto máximo: ___ g'."
                    )

            except Exception as e:
                st.error(f"Erro ao chamar o LLM: {e}")

    with open(tabela_file, "rb") as f:
        st.download_button(
            "Download: Tabela Estruturada (.csv)",
            data=f,
            file_name=os.path.basename(tabela_file),
            mime="text/csv"
        )
