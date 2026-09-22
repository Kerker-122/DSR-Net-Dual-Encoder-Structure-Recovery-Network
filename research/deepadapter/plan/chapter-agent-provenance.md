# Chapter provenance

The invoked skill requested fresh chapter agents. Three were dispatched with separate persistent task packets and write ownership. All ended with usage-limit errors. The Introduction file existed and was read/reviewed by the controller. No subagent falsely recorded as completed.

The user's subsequent instruction was to continue the same drafting task. The controller completed the remaining chapters locally and reviewed all five. This deviates from the skill's ideal separate-agent workflow; the deviation is disclosed rather than hidden or reported as a multi-agent quality pass.

| File | Owner / attempt | Inputs | Review status |
|---|---|---|---|
| chapters/01_Introduction.md | /root/introduction wrote file then errored; controller reviewed | packet 01, evidence map, blueprint | Accepted as first draft with 2 data gaps; not final manuscript |
| chapters/02_Related_Work.md | /root/related_work errored before file; /root authored | packet 02, source map, blueprint | Accepted as first draft; prior work reference pending |
| chapters/03_Method.md | /root/method errored before file; /root authored | packet 03, prior ICCS code, user architecture | Accepted as provisional methodology; actual new code/loss pending |
| chapters/04_Experiments.md | /root authored after collaboration failure | packet 04, protocol, user report | Provisional reporting/protocol draft; no verified numeric results |
| chapters/05_Conclusion.md | /root authored after collaboration failure | packet 05, four drafted chapters | Provisional conclusion, result claim explicitly marked |
