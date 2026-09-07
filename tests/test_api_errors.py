import json
import unittest
from unittest.mock import patch

import requests

from src import api, models


OUTAGE = "The AniList API has been temporarily disabled due to severe stability issues."


def response(status, payload):
    result = requests.Response()
    result.status_code = status
    result.url = api.API_URL
    result._content = json.dumps(payload).encode()
    return result


class AniListErrorTests(unittest.TestCase):
    def setUp(self):
        self.post = self.enterContext(patch.object(api._HTTP, "post"))
        self.sleep = self.enterContext(patch.object(api.time, "sleep"))
        self.enterContext(patch.object(api, "_respect_anilist_rate_limit"))
        api._set_last_api_error("")
        self.addCleanup(api._set_last_api_error, "")

    def test_outage_explains_the_reason_without_retrying(self):
        self.post.return_value = response(403, {"errors": [{"message": OUTAGE}], "data": None})
        with self.assertRaisesRegex(api.AniListAccessError, "temporarily disabled") as raised:
            api.fetch_with_retry("query { __typename }", {})
        self.assertIn("try again later", str(raised.exception))
        self.assertEqual(api.get_last_api_error(), str(raised.exception))
        self.post.assert_called_once()
        self.sleep.assert_not_called()

    def test_non_json_forbidden_has_a_readable_error(self):
        result = response(403, None)
        result._content = b"<html>Forbidden</html>"
        self.post.return_value = result
        with self.assertRaisesRegex(api.AniListAccessError, r"refused access.*403"):
            api.fetch_with_retry("query { __typename }", {})
        self.post.assert_called_once()
        self.sleep.assert_not_called()

    def test_other_access_errors_keep_the_actual_reason(self):
        self.post.return_value = response(403, {"errors": [{"message": "Access denied for this request."}]})
        with self.assertRaisesRegex(api.AniListAccessError, "Access denied for this request") as raised:
            api.fetch_with_retry("query { __typename }", {})
        self.assertNotIn("stability", str(raised.exception))

    def test_success_preserves_request_and_data_and_clears_previous_error(self):
        data = {"User": {"id": 123, "name": "example"}}
        query, variables = "query ($name: String) { User(name: $name) { id name } }", {"name": "example"}
        api._set_last_api_error("Previous outage")
        self.post.return_value = response(200, {"data": data})
        self.assertEqual(api.fetch_with_retry(query, variables), data)
        self.assertEqual(self.post.call_args.kwargs["json"], {"query": query, "variables": variables})
        self.assertEqual(api.get_last_api_error(), "")

    def test_transient_server_error_still_retries(self):
        self.post.side_effect = [response(503, {}), response(200, {"data": {"ok": True}})]
        self.assertEqual(api.fetch_with_retry("query", {}), {"ok": True})
        self.assertEqual(self.post.call_count, 2)
        self.sleep.assert_called_once_with(1)

    def test_rate_limit_still_respects_retry_after(self):
        limited = response(429, {})
        limited.headers["Retry-After"] = "5"
        self.post.side_effect = [limited, response(200, {"data": {"ok": True}})]
        self.assertEqual(api.fetch_with_retry("query", {}), {"ok": True})
        self.sleep.assert_called_once_with(5)

    def test_graphql_validation_error_keeps_existing_behavior(self):
        self.post.return_value = response(200, {"errors": [{"message": "User not found"}]})
        with self.assertRaisesRegex(ValueError, "GraphQL Error: User not found"):
            api.fetch_with_retry("query", {})
        self.post.assert_called_once()

    def test_outage_does_not_train_or_overwrite_anime_or_manga_artifacts(self):
        for train in (models.train_and_evaluate_all_models, models.train_and_evaluate_all_manga_models):
            with self.subTest(train=train.__name__):
                self.post.return_value = response(403, {"errors": [{"message": OUTAGE}]})
                with patch.object(models, "_core_train_pipeline") as fit, patch.object(models.joblib, "dump") as save:
                    result = train("outage_regression_test")
                self.assertEqual(result["status"], "error")
                self.assertIn("AniList has temporarily disabled", result["message"])
                self.assertNotIn("HTTPError", result["message"])
                fit.assert_not_called()
                save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
