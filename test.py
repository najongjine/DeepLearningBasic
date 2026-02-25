import os
from groq import Groq

# https://console.groq.com/docs/models

# API 키를 환경 변수에서 불러오거나 직접 문자열로 넣으셔도 됩니다.
client = Groq(
    api_key="[ENCRYPTION_KEY]", 
)

def summarize_text(content):
    # Groq에서 지원하는 빠르고 가벼운 모델 (예: Llama 3 8B)
    model_name = "llama-3.1-8b-instant" 
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "당신은 AI 비서입니다."
            },
            {
                "role": "user",
                "content": f"다음 텍스트를 요약해 줘:\n\n{content}"
            }
        ],
        model=model_name,
        temperature=0.3, # 요약 작업이므로 환각을 줄이기 위해 온도를 낮게 설정
        max_tokens=512,
    )
    
    return chat_completion.choices[0].message.content

# 테스트 실행
sample_text = """
 such as OpenAI, but every exec seemed quite dialed in. You'd see gdb, sama, kw, mark, dane, et al chime in regularly on Slack. There are no absentee leaders.

Code
OpenAI uses a giant monorepo which is ~mostly Python (though there is a growing set of Rust services and a handful of Golang services sprinkled in for things like network proxies). This creates a lot of strange-looking code because there are so many ways you can write Python. You will encounter both libraries designed for scale from 10y Google veterans as well as throwaway Jupyter notebooks from newly-minted PhDs. Pretty much everything operates around FastAPI to create APIs and Pydantic for validation. But there aren't style guides enforced writ-large.

OpenAI runs everything on Azure. What's funny about this is there are exactly three services that I would consider trustworthy: Azure Kubernetes Service, CosmosDB (Azure's document storage), and BlobStore. There's no true equivalents of Dynamo, Spanner, Bigtable, Bigquery Kinesis or Aurora. It's a bit rarer to think a lot in auto-scaling units. The IAM implementations tend to be way more limited than what you might get from an AWS. And there's a strong bias to implement in-house.

When it comes to personnel (at least in eng), there's a very significant Meta → OpenAI pipeline. In many ways, OpenAI resembles early Meta: a blockbuster consumer app, nascent infra, and a desire to move really quickly. Most of the infra talent I've seen brought over from Meta + Instagram has been quite strong.

Put these things together, and you see a lot of core parts of infra that feel reminiscent of Meta. There was an in-house reimplementation of TAO. An effort to consolidate auth identity at the edge. And I'm sure a number of others I don't know about.

Chat runs really deep. Since ChatGPT took off, a lot of the codebase is structured around the idea of chat messages and conversations. These primitives are so baked at this point, you should probably ignore them at your own peril. We did deviate from them a bit in Codex (leaning more into learnings from the responses API), but we leveraged a lot of prior art.

Code wins. Rather than having some central architecture or planning committee, decisions are typically made by whichever team plans to do the work. The result is that there's a strong bias for action, and often a number of duplicate parts of the codebase. I must've seen half a dozen libraries for things like queue management or agent loops.

There were a few areas where having a rapidly scaled eng team and not a lot of tooling created issues. sa-server (the backend monolith) was a bit of a dumping ground. CI broke a lot more frequently than you might expect on master. Test cases even running in parallel and factoring in a subset of dependencies could take ~30m to run on GPUs. These weren't unsolvable problems, but it's a good reminder that these sorts of problems exist everywhere, and they are likely to get worse when you scale super quickly. To the credit of the internal teams, there's a lot of focus going into improving this story.

Other things I learned
What a big consumer brand looks like. I hadn't really internalized this until we started working on Codex. Everything is measured in terms of 'pro subs'. Even for a product like Codex, we thought of the onboarding primarily related to individual usage rather than teams. It broke my brain a bit, coming from predominantly a B2B / enterprise background. You flip a switch and you get traffic from day 1.

How large models are trained (at a high-level). There's a spectrum from "experimentation" to "engineering". Most ideas start out as small-scale experiments. If the results look promising, they then get incorporated into a bigger run. Experimentation is as much about tweaking the core algorithms as it is tweaking the data mix and carefully studying the results. On the large end, doing a big run almost looks like giant distributed systems engineering. There will be weird edge cases and things you didn't expect. It's up to you to debug them.

How to do GPU-math. We had to forecast out the load capacity requirements as part of the Codex launch, and doing this was the first time I'd really spent benchmarking any GPUs. The gist is that you should actually start from the latency requirements you need (overall latency, # of tokens, time-to-first-token) vs doing bottoms-up analysis on what a GPU can support. Every new model iteration can change the load patterns wildly.

How to work in a large Python codebase. Segment was a combination of both microservices, and was mostly Golang and Typescript. We didn't really have the breadth of code that OpenAI does. I learned a lot about how to scale a codebase based upon the number of developers contributing to it. You have to put in a lot more guardrails for things like "works by default", "keep master clean", and "hard to misuse".

Launching Codex
A big part of my last three months at OpenAI was launching Codex. It's unquestionably one of the highlights of my career.

To set the stage, back in November 2024, OpenAI had set a 2025 goal to launch a coding agent. By February 2025 we had a few internal tools floating around which were using the models to great effect. And we were feeling the pressure to launch a coding-specific agent. Clearly the models had gotten to the point where they were getting really useful for coding (seeing the new explosion of vibe-coding tools in the market).

I returned early from my paternity leave to help participate in the Codex launch. A week after I returned, we had a (slightly chaotic) merger of two teams, and began a mad-dash sprint. From start (the first lines of code written) to finish, the whole product was built in just 7 weeks.

The Codex sprint was probably the hardest I've worked in nearly a decade. Most nights were up until 11 or midnight. Waking up to a newborn at 5:30 every morning. Heading to the office again at 7a. Working most weekends. We all pushed hard as a team, because every week counted. It reminded me of being back at YC.

It's hard to overstate how incredible this level of pace was. I haven't seen organizations large or small go from an idea to a fully launched + freely available product in such a short window. The scope wasn't small either; we built a container runtime, made optimizations on repo downloading, fine-tuned a custom model to deal with code edits, handled all manner of git operations, introduced a completely new surface area, enabled internet access, and ended up with a product that was generally a delight to use. 4

Say what you will, OpenAI still has that launching spirit. 5

The good news is that the right people can make magic happen. We were a senior team of ~8 engineers, ~4 researchers, 2 designers, 2 GTM and a PM. Had we not had that group, I think we would've failed. Nobody needed much direction, but we did need a decent amount of coordination. If you get the chance to work with anyone on the Codex team, know that every one of them is fantastic.

The night before launch, five of us stayed up until 4a trying to deploy the main monolith (a multi-hour affair). Then it was back to the office for the 8a launch announcement and livestream. We turned on the flags, and started to see see the traffic pour in. I've never seen a product get so much immediate uptick just from appearing in a left-hand sidebar, but that's the power of ChatGPT.

In terms of the product shape, we settled on a form factor which was entirely asynchronous. Unlike tools like Cursor (at the time, it now supports a similar mode) or Claude Code, we aimed to allow users to kick off tasks and let the agent run in its own environment. Our bet was in the end-game, users should treat a coding agent like a co-worker: they'd send messages to the agent, it gets some time to do its work, and then it comes back with a PR.

This was a bit of a gamble: we're in a slightly weird state today where the models are good, but not great. They can work for minutes at a time, but not yet hours. Users have widely varying degrees of trust in the models capabilities. And we're not even clear what the true capabilities of the models are.

Over the long arc of time, I do believe most programming will look more like Codex. In the meantime, it's going to be interesting to see how all the products unfold.

Codex (maybe unsurprisingly) is really good at working in a large codebase, understanding how to navigate it. The biggest differentiator I've seen vs other tools is the ability to kick off multiple tasks at once and compare their output.

I recently saw that there are public numbers comparing the PRs made by different LLM agents. Just at the public numbers, Codex has generated 630,000 PRs. That's about 78k public PRs per engineer in the 53 days since launch (you can make your own guesses about the multiple of private PRs). I'm not sure I've ever worked on something so impactful in my life.

Parting thoughts
Truth be told, I was originally apprehensive about joining OpenAI. I wasn't sure what it would be like to sacrifice my freedom, to have a boss, to be a much smaller piece of a much larger machine. I kept it fairly low-key that I had joined, just in case it wasn't the right fit.

I did want to get three things from the experience...

to build intuition for how the models were trained and where the capabilities were going
to work with and learn from amazing people
to launch a great product
In reflecting on the year, I think it was one of the best moves I've ever made. It's hard to imagine learning more anywhere else.

If you're a founder and feeling like your startup really isn't going anywhere, you should either 1) deeply re-assess how you can take more shots on goal or 2) go join one of the big labs. Right now is an incredible time to build. But it's also an incredible time to peer into where the future is headed.

As I see it, the path to AGI is a three-horse race right now: OpenAI, Anthropic, and Google. Each of these organizations are going to take a different path to get there based upon their DNA (consumer vs business vs rock-solid-infra + data). 6 Working at any of them will be an eye-opening experience.

Thank you to Leah for being incredibly supportive and taking the majority of the childcare throughout the late nights. Thanks to PW, GDB, and Rizzo for giving me a shot. Thanks to the SA teammates for teaching me the ropes: Andrew, Anup, Bill, Jeremy, Kwaz, Ming, Simon, Tony, and Val. And thanks for the Codex core team for giving me the ride of a lifetime: Albin, AE, Andrey, Bryan, Channing, DavidK, Gabe, Gladstone, Hanson, Joey, Josh, Katy, KevinT, Max, Sabrina, SQ, Tibo, TZ and Will. I'll never forget this sprint.

Wham.

It's easy to try and read into a lot of drama whenever there's a departing leader, but I would chalk ~70% of them up to this fact alone. ↩

I do think we're in a slight phase change here. There's a lot of senior leadership hires being made from outside the company. I'm generally in favor of this, I think the company benefits a lot from infusing new external DNA. ↩

I get the sense that scaling the fastest growing consumer product ever tends to build a lot of muscle. ↩

Of course, we were also standing on the shoulders of giants. The CaaS team, core RL teams, human data, and general applied infra made this all possible. ↩

We kept it going too. ↩

We saw some big hires at Meta a few weeks ago. xAI launched Grok 4 which performs well on benchmarks. Mira and Ilya both have great talent. Maybe that will change things (the people are good). They have some catching up to do. ↩
"""
result = summarize_text(sample_text)
print("요약 결과:\n", result)