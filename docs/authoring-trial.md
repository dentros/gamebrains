# Authoring trial: instructions for participants

Thank you for doing this. It should take **60 to 120 minutes**, and you are free to stop at any
point.

You are not being tested. **We are testing the documentation.** If you get stuck, that is the
result we are looking for, and a trial where you give up after forty minutes is more useful to us
than one where you succeed by guessing. Please do not try to make us look good.

**You will not be named against anything you write here.** If any of this is quoted or reported, it
appears as "P1", "P2" and so on, described by background rather than by name. We say this now rather
than only in the consent section at the end, because it is the reason you can afford to write "I
gave up", "this made no sense to me" or "I copied an existing file because the guide was not
helping". Those are the sentences that make the exercise worth doing, and a log written by someone
who feels identifiable does not contain them.

If you would *like* to be credited, there is a separate box for that at the end. Being thanked by
name and having your log attributed to you are different things, and you can have the first without
the second.

---

## What we ask you to do

1. Clone `https://github.com/dentros/gamebrains` and get the test suite running.
2. Read `docs/adding-a-game.md` and implement **one new game** of your choice against it.
3. Read `docs/adding-an-agent.md` and implement **one new agent** of your choice against it.
4. Run your game and your agent together in a single match.
5. Fill in the log at the bottom of this file and send it back.

Your game and agent can be as simple as you like. A trivially simple one that works teaches us as
much as an elaborate one.

## Ground rules that matter for the result

- **Use only the repository's own documentation, plus anything you would normally use** (search
  engines, an AI assistant, existing code in the repo). Please record it if you do.
- **Do not ask us questions during the trial.** If you would normally have asked a colleague, write
  the question down in the log instead and carry on however you see fit. Those questions are the
  single most valuable thing you can give us, and if we answer them the gap disappears from our
  data.
- **Log stuck-points as they happen**, not afterwards. A note written twenty minutes later has lost
  what you actually believed at the time.
- **Record real clock times**, not estimates.

## Consent

Fill in the last section. In short: we would like to quote what you write, anonymously, in a
software-engineering paper about the platform. You can decline that and still take part, and you can
withdraw afterwards.

---

# Trial log

Copy everything below into your reply, or edit this file and send it back.

## About you

| | |
|---|---|
| Years of Python experience | |
| Have you used a multi-agent or RL framework before? Which? | |
| Have you seen this codebase before today? | |
| Operating system and Python version | |

**Which hat are you wearing today?** Tick whichever fits; both is a fine answer.

- [ ] **Developer.** You are judging the interface: is the contract clear, do the error messages
      help, is anything missing that you needed.
- [ ] **Researcher.** You are judging whether you could do your own work on this: could you add
      your own measure, set up a comparison you would trust, get results you would publish.
- [ ] Both.

We ask because the two notice different things and we want to know which lens produced which
comment, not because one is more welcome than the other.

## Timings

Write down the clock time at each point. Leave a row blank if you did not reach it.

| Stage | Time started | Time finished | Gave up? |
|---|---|---|---|
| Clone and get the test suite passing | | | |
| Read `adding-a-game.md` | | | |
| Implement your game | | | |
| Get your game's own test passing | | | |
| Read `adding-an-agent.md` | | | |
| Implement your agent | | | |
| Run game and agent together in one match | | | |

## Stuck-points

One block per time you were blocked for more than about five minutes. Add as many as you need.

### Stuck-point 1

- **What I was trying to do:**
- **What I expected to happen:**
- **What actually happened** (paste any error message in full):
- **What I tried, in order:**
- **What unblocked me, or why I stopped trying:**
- **Roughly how long I was stuck:**

### Stuck-point 2

- **What I was trying to do:**
- **What I expected to happen:**
- **What actually happened:**
- **What I tried, in order:**
- **What unblocked me, or why I stopped trying:**
- **Roughly how long I was stuck:**

## Questions you would have asked a colleague

Write them as you would have asked them. Do not resolve them for us.

1.
2.
3.

## Specific things we want to know

Please answer from what you did, not from what seems polite.

1. **Which parts of a guide did you re-read?** Naming a section is enough.

2. **Did you copy an existing file as a starting point instead of following the guide?** Which one,
   and at what moment did you decide to?

3. `semantics` is a required declaration on an agent, `"index-agnostic"` or `"role-bound"`.
   - Which did you choose, and how long did the choice take?
   - Did you understand *why* it is required? Answer honestly if not.
   - Did you hit `SemanticBindingError`? If so, paste the message. Did it tell you what to do next?

4. **Did anything refuse to run and tell you why?** Was the message useful, useless, or misleading?

5. **Did anything run when you expected it to fail**, or produce a result you did not trust?

6. **What did you assume that turned out to be wrong?** This one is worth more than the rest.

7. **If a colleague asked you tomorrow to add a second game, how long would you say it would take?**

8. **What would you delete from the guides?** We ask this because everyone adds and nobody removes.

## Anything else

Free text. Complaints are welcome and are more useful than praise.

## Consent

- [ ] I agree that anonymised excerpts of this log may be quoted in an academic publication about
      the GameBrains platform.
- [ ] I would like to be acknowledged by name in that publication.
      Name as you wish it to appear: ______________________
- [ ] I would prefer my log be used only in aggregate, with nothing quoted.

You may withdraw this consent at any time before publication by contacting the authors, with no
explanation needed.

---

## Notes for whoever runs the trial

Keep this section out of what you send participants.

- **Do not answer questions during a trial.** Every answer given is a data point destroyed, and the
  temptation is strongest for the gaps that matter most.
- **Send the participant this file and nothing else.** No verbal walkthrough, no "just so you
  know", no pre-emptive warnings about known rough edges. If a rough edge needs a warning, that
  warning belongs in the guide.
- **Run participants independently**, and do not fix the documentation between them until the round
  is finished. Fixing as you go means no two logs describe the same artefact and none of them can be
  compared.
- **A participant who gives up is a completed trial**, not a failed one. Record the point of
  abandonment and thank them properly.
- **Scale.** A handful of participants supports an experience report: what happened to specific
  people meeting this contract for the first time. It does not support a claim about the interface
  in general. If that stronger claim is wanted, the design changes: recruitment, a real instrument,
  a pre-registered analysis, and institutional ethics approval. Do not let a small round quietly
  grow into a claim it cannot carry.

### Recruitment, recorded as it happens

Fill this in while recruiting, not afterwards. Whatever the answers are, they go in the paper: a
disclosed recruitment relationship is a limitation, an undisclosed one is a finding.

| | |
|---|---|
| Number approached / number who took part | |
| How they were found | |
| Relationship to the authors | |
| Who collected the completed logs | |
| Was any participant in a position of dependency on an author? | |

**On dependency, because it is the trap here.** A participant who is an author's student cannot
freely decline, and cannot freely write that the documentation is bad. Consent forms do not fix
this; they only record it. In order of preference:

1. Recruit outside any dependent relationship: colleagues, doctoral students from another group,
   external contacts. This costs nothing extra and removes the problem rather than managing it.
2. If participants must be students, have someone else collect the logs and pass them on
   anonymised, or run the trial after any assessment they are involved in has concluded.
3. Authors recruiting their own students and reading identified logs is the version a reviewer
   experienced in empirical software engineering will notice and name.

**Mix the roles.** A developer will find ambiguities in the contract and unhelpful error messages.
A researcher will ask how to add their own measure and how to make a comparison they would trust.
Those are different failures and one participant type will not surface both.
