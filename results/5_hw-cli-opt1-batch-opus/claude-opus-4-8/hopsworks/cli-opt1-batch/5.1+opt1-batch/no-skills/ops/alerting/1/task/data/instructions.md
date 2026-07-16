# Flaky job + failure alert

`failing_job.py` ALWAYS exits 1 (it raises a seeded error, ERR-685204). That is intentional: the goal is to verify that a FAILURE of the job raises an alert. Configure the platform's alerting mechanism for the job's failures — name or mention `flaky606527` in the alert so it is identifiable.
