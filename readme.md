Post
http://127.0.0.1:5000/load_blog
{
  "session_id": "test-session-1",
  "blog_content": "The United States stands as a dynamic beacon of innovation, diversity, and resilience. From Silicon Valley’s cutting-edge technology to the vibrant cultural mosaic of its cities, it constantly reinvents itself. Rooted in principles of freedom and democracy, the nation faces modern challenges with unyielding spirit—embracing change, pushing boundaries, and striving for inclusion. America’s strength lies in its people’s dreams and determination, crafting a future where opportunity is boundless, creativity flourishes, and every voice can shape the story. In a rapidly evolving world, the USA remains a powerful symbol of progress and hope."
}

{
    "message": "Blog content loaded and embeddings created for session 'test-session-1'."
}

Post
http://127.0.0.1:5000/ask
{
  "session_id": "test-session-1",
  "question": "Principles of USA?"
}
{
    "answer": "Freedom and democracy",
    "chat_history": [
        {
            "additional_kwargs": {},
            "content": "Principles of USA?",
            "example": false,
            "id": null,
            "name": null,
            "response_metadata": {},
            "type": "human"
        },
        {
            "additional_kwargs": {},
            "content": "Freedom and democracy",
            "example": false,
            "id": null,
            "invalid_tool_calls": [],
            "name": null,
            "response_metadata": {},
            "tool_calls": [],
            "type": "ai",
            "usage_metadata": null
        }
    ]
}