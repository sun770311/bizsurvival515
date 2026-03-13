from __future__ import annotations

import streamlit as st

from utils.ui_styles import apply_shared_styles


st.set_page_config(page_title="Findings", layout="wide")
apply_shared_styles()

st.markdown(
    """
    <style>
    .section-card {
        background: linear-gradient(
            135deg,
            rgba(127, 255, 212, 0.22) 0%,
            rgba(255, 255, 120, 0.18) 45%,
            rgba(11, 102, 35, 0.20) 100%
        );
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 18px;
        padding: 14px 14px;
        margin: 6px 0 16px 0;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
    }

    .section-card h3 {
        margin: 0;
        padding: 0;
        line-height: 1.2;
        font-size: 1.12rem;
        font-weight: 600;
        color: white;
    }

    .info-card {
        background: linear-gradient(
            135deg,
            rgba(127, 255, 212, 0.22) 0%,
            rgba(255, 255, 120, 0.18) 45%,
            rgba(11, 102, 35, 0.20) 100%
        );
        border: 1px solid rgba(255,255,255,0.28);
        border-radius: 18px;
        padding: 18px 18px 16px 18px;
        min-height: 220px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
    }

    .info-card a {
        color: #9be7c4 !important;
        text-decoration: none;
        font-weight: 600;
        margin-left: 6px;
    }

    .info-card a:hover {
        text-decoration: underline;
    }

    .info-card h4 {
        margin: 0 0 10px 0;
        font-size: 1.02rem;
        font-weight: 600;
        color: white;
    }

    .info-card p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.6;
        color: rgba(255,255,255,0.92);
    }

    .glass-banner {
        margin-top: 8px;
        margin-bottom: 18px;
        padding: 14px 18px;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            rgba(127,255,212,0.30) 0%,
            rgba(255,255,140,0.25) 50%,
            rgba(11,102,35,0.30) 100%
        );
        border: 1px solid rgba(255,255,255,0.35);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        color: white;
        font-size: 15px;
        line-height: 1.55;
        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
    }

    .warn-banner {
        margin-top: 4px;
        margin-bottom: 18px;
        padding: 14px 18px;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            rgba(255,80,40,0.28) 0%,
            rgba(255,150,40,0.25) 45%,
            rgba(255,210,80,0.28) 100%
        );
        border: 1px solid rgba(255,200,120,0.35);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        color: white;
        font-size: 14.5px;
        line-height: 1.55;
        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
    }

    .section-divider {
        height: 1px;
        margin: 34px 0 24px 0;
        background: linear-gradient(
            90deg,
            rgba(127,255,212,0.6),
            rgba(255,255,140,0.6),
            rgba(11,102,35,0.6)
        );
        opacity: 0.7;
    }

    .about-box {
        background: linear-gradient(
            135deg,
            rgba(127,255,212,0.30) 0%,
            rgba(255,255,140,0.25) 50%,
            rgba(11,102,35,0.30) 100%
        );

        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 18px;

        padding: 18px 20px;

        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);

        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
    }

    .about-box a {
        color: #9be7c4 !important;
        text-decoration: none;
        font-weight: 600;
    }

    .about-box a:hover {
        text-decoration: underline;
    }

    .dataset-gap {
        height: 34px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Findings")

discussion_tab, datasets_tab, about_tab = st.tabs(
    ["Discussion of Results", "Datasets Used", "About"]
)

with discussion_tab:
    st.markdown(
        """
    Across all three models, the main takeaway is not that business survival can be predicted perfectly,
    but that the data does contain useful structure. Neighborhood conditions, business type, licensing,
    and complaint activity all appear to contribute some predictive signal.
    """
    )

    st.markdown(
        """
        <div class="section-card">
            <h3>Logistic Regression: Main Interpretation</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        The logistic regression model was designed to answer a direct classification question:
        whether a business survives at least 3 years. Its output is easier to interpret than
        the Cox models because it produces a survival probability directly. Businesses that actually survived at least 36 months had an average predicted survival
        probability of **0.5396**, while businesses that closed earlier had an average predicted
        survival probability of **0.4602**. This gap is not huge, but it is meaningful. It shows
        that the model tends to assign higher probabilities to businesses that truly survived and
        lower probabilities to businesses that did not.

        At the same time, these results should be interpreted carefully. Entrepreneurship is very
        difficult to predict because businesses are affected by many factors that are not fully
        captured in licensing records, neighborhood conditions, or 311 complaint data. Ownership
        changes, financing, management quality, competition, consumer demand, regulation, and random
        shocks can all affect closure risk. Risk and success also do not follow a simple linear
        relationship, so a logistic regression model is naturally limited in how much of the real
        process it can represent.

        Even with those limitations, the model performed better than random chance in separating
        survivors from non-survivors. That is encouraging because it suggests the available features
        are not just noise: they do contain some statistically useful signal, even if that signal
        is incomplete.
        """
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">
            <h3>Comparing Logistic Regression and the Cox Models</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        The three models overlap in a meaningful way, which is reassuring. Several business categories
        appear repeatedly among influential predictors, including electronics-related businesses,
        pedicab-related businesses, home improvement contractors, garage and parking lot businesses,
        and some secondhand or specialty service categories. This suggests that baseline business type
        carries real information about business risk, regardless of the modeling framework.

        The models also emphasize different parts of the problem.

        The **logistic regression** model highlights a mix of geography, business categories, active
        license count, and a small number of complaint-related predictors. That makes sense for a model
        trying to draw a broad boundary between businesses that survive 3 years and those that do not. The **standard Cox model** overlaps with logistic regression on several business categories,
        but its strongest signals lean more toward baseline business-type effects. This is consistent
        with how the model is built: it uses only the initial state of a business. At that early point,
        complaint history is still limited, so the model naturally relies more heavily on business
        category and starting characteristics. The **time-varying Cox model** looks different in an informative way. Its strongest hazard-increasing
        predictors are complaint types rather than business categories. This is probably because, unlike the
        standard Cox model, the time-varying model has a chance to observe conditions after a business has
        operated long enough for complaints to accumulate. Once the model is allowed to update over time,
        complaint patterns become much more informative.

        This difference between the two Cox models is one of the most important findings in the project.
        The standard Cox model mostly reflects **who the business is at the beginning**, while the time-varying
        Cox model better reflects **what is happening around the business as time passes**.
        """
    )

    st.markdown(
        """
        <div class="warn-banner">
            <b>Interpretation note:</b> These predictors should be read as associations learned from historical data,
            not as causal claims. A complaint type or business category appearing as influential does not mean it
            directly causes closure; it means it helped the model distinguish different risk patterns in this dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">
            <h3>Overall Conclusion</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Overall, the models tell a coherent story. Baseline business type matters, geography matters,
        and evolving neighborhood or service-request conditions matter too. The logistic regression model shows that even a relatively simple probability model can recover
        useful separation between businesses that survive and businesses that close earlier. The standard
        Cox model shows that baseline business characteristics carry meaningful long-run risk information.
        The time-varying Cox model shows that once conditions are allowed to change over time, complaint
        activity becomes a much stronger signal. These findings support the idea that business survival is not random, but it is
        also not easily predictable. The best interpretation is that the data offers **partial visibility**
        into business risk, not a full explanation of business success or failure. We are glad that the
        models were able to recover at least some significant signal from the data, because predicting
        entrepreneurship is inherently difficult and anything can happen at any time.
        """
    )

with datasets_tab:
    top_left, top_right = st.columns(2)

    with top_left:
        st.markdown(
            """
            <div class="info-card">
                <h4>
                    Dataset 1: Issued Licenses
                    <a href="https://data.cityofnewyork.us/Business/Issued-Licenses/w7w3-xahh/about_data"
                       target="_blank" rel="noopener noreferrer">[Link]</a>
                </h4>
                <p>
                    This dataset contains licensed businesses operating in New York City. 
                    It is updated weekly and includes key fields such as business identifiers, business name,
                    category, license status, address, borough, and geographic coordinates.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        st.markdown(
            """
            <div class="info-card">
                <h4>
                    Dataset 2: 311 Service Requests
                    <a href="https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data"
                       target="_blank" rel="noopener noreferrer">[Link]</a>
                </h4>
                <p>
                    This dataset contains over 20 million rows of resident-submitted complaints and service requests. 
                    It covers data from 2020 to the present and is updated daily. This dataset is much larger and includes 
                    complaint type, request date, and location coordinates. Example request categories include 
                    noise, sanitation, street conditions, and maintenance-related issues.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="dataset-gap"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-banner">
            <b>Join logic:</b> Complaints were joined to businesses using spatial location and time alignment.
            The resulting joined dataset contains <b>one row per business per month</b>, which supports both
            baseline modeling and time-varying survival analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Why These Datasets Matter")
    st.markdown(
        """
        The license dataset provides the core business reference frame: what the business is, where it is,
        and whether it appears active in the city's licensing records. The 311 dataset adds a dynamic view
        of local conditions and complaints around those businesses.

        Together, these two datasets support the project's central idea: business outcomes may be partly
        explained not only by business category and location, but also by the surrounding complaint and
        service-request environment. The monthly join structure is especially important because it lets the
        project compare fixed baseline models with models that update risk over time.
        """
    )

    st.markdown("### Computational Note")
    st.markdown(
        """
        The full joined dataset was run in **Google Colab using an NVIDIA G4 GPU**. The model artifacts
        and results currently stored in the GitHub repository were produced using a **subset of the
        311 service requests data**, which made the workflow easier to test and share within the
        project repository.
        """
    )

    st.markdown("### Dataset Caveats")
    st.markdown(
        """
        These datasets are useful, but they are not a complete representation of entrepreneurship in NYC.
        Licensing data does not capture every operational detail of a business, and 311 complaints reflect
        only reported issues rather than the full reality of neighborhood conditions. Spatial matching and
        timing alignment are also approximations, which means the joined dataset is informative but imperfect.
        """
    )

    st.markdown("### Data Source Citation")

    st.markdown(
        """
        Both datasets used in this project were obtained from
        <a href="https://opendata.cityofnewyork.us/" target="_blank">
        NYC Open Data</a>.

        - Issued Licenses dataset:  
        <a href="https://data.cityofnewyork.us/Business/Issued-Licenses/w7w3-xahh/about_data" target="_blank">
        https://data.cityofnewyork.us/Business/Issued-Licenses/w7w3-xahh/about_data</a>

        - 311 Service Requests dataset:  
        <a href="https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data" target="_blank">
        https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data</a>
        """,
        unsafe_allow_html=True,
    )

with about_tab:
    st.markdown(
        """
        <div class="about-box">
            <h3 style="margin-top:0;">Project Overview</h3>
            <p>
                <b>Repository:</b>
                <a href="https://github.com/sun770311/bizsurvival515" target="_blank" rel="noopener noreferrer">
                    github.com/sun770311/bizsurvival515
                </a>
            </p>
            <p>
                If you find this project interesting or useful, please consider starring the repository on GitHub! ⭐ ⭐ 
            </p>
            <p>
                <b>Authors:</b> Hannah Sun, Juan Pablo Reyes Martinez, Sreeraj Parakkat, Pavankumar Suresh, Aaron Lee
            </p>
            <p>
                A University of Washington MSDS DATA 515 project.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
