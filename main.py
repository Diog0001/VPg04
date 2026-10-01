"""Telegram-бот для RAG-агента на PyTelegramBotAPI."""

from __future__ import annotations

from collections import defaultdict, deque

import telebot
from dotenv import load_dotenv
from telebot.types import KeyboardButton, ReplyKeyboardMarkup

from rag_agent import RAGAgent, _require_env, extract_urls

load_dotenv()
load_dotenv(".env.example", override=False)

TELEGRAM_LIMIT = 4000

BTN_CAT_KNOWLEDGE = "База знаний"
BTN_CAT_MEMORY = "Память"
BTN_HELP = "Справка"
BTN_BACK = "Назад"
BTN_SEARCH = "Поиск по базе"
BTN_ADD_TEXT = "Добавить текст"
BTN_ADD_URL = "Добавить статью по URL"
BTN_REMEMBER = "Запомнить обо мне"
BTN_CANCEL = "Отмена"

HELP_TEXT = (
    "Я умный помощник с векторной памятью. Обычное сообщение — это вопрос ко мне.\n\n"
    "Категории кнопок:\n\n"
    "База знаний\n"
    "• Поиск по базе — найти фрагменты без генерации ответа\n"
    "• Добавить текст — сохранить заметку\n"
    "• Добавить статью по URL — прочитать HTML-страницу и положить в базу\n\n"
    "Память\n"
    "• Запомнить обо мне — сохранить факт о вас\n\n"
    "Справка — это сообщение. Назад — в главное меню."
)


def main_keyboard() -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.row(KeyboardButton(BTN_CAT_KNOWLEDGE), KeyboardButton(BTN_CAT_MEMORY))
    keyboard.row(KeyboardButton(BTN_HELP))
    return keyboard


def knowledge_keyboard() -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.row(KeyboardButton(BTN_SEARCH))
    keyboard.row(KeyboardButton(BTN_ADD_TEXT), KeyboardButton(BTN_ADD_URL))
    keyboard.row(KeyboardButton(BTN_BACK))
    return keyboard


def memory_keyboard() -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.row(KeyboardButton(BTN_REMEMBER))
    keyboard.row(KeyboardButton(BTN_BACK))
    return keyboard


def cancel_keyboard() -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.row(KeyboardButton(BTN_CANCEL))
    return keyboard


def _chunks(text: str, size: int = TELEGRAM_LIMIT) -> list[str]:
    text = text or "Пустой ответ."
    return [text[index : index + size] for index in range(0, len(text), size)]


def format_search_results(docs) -> str:
    if not docs:
        return "По этому запросу в базе ничего не нашёл."
    parts = [f"Нашёл фрагментов: {len(docs)}"]
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        kind = doc.metadata.get("kind", "unknown")
        body = " ".join(doc.page_content.split())
        if len(body) > 500:
            body = body[:500] + "…"
        parts.append(f"{index}. [{kind}] {source}\n{body}")
    return "\n\n".join(parts)


def send_long(
    bot: telebot.TeleBot,
    chat_id: int,
    text: str,
    reply_to: int | None = None,
    reply_markup: ReplyKeyboardMarkup | None = None,
) -> None:
    parts = _chunks(text)
    for index, part in enumerate(parts):
        bot.send_message(
            chat_id,
            part,
            reply_to_message_id=reply_to if index == 0 else None,
            reply_markup=reply_markup if index == len(parts) - 1 else None,
        )


def run_bot() -> None:
    token = _require_env("TELEGRAM_BOT_TOKEN")
    print("Подключаю RAG-агент…")
    agent = RAGAgent()
    bot = telebot.TeleBot(token, parse_mode=None)
    pending: dict[int, str] = {}
    menus: dict[int, str] = {}
    histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=12))

    def user_id_of(message: telebot.types.Message) -> int:
        return message.from_user.id

    def section_keyboard(user_id: int) -> ReplyKeyboardMarkup:
        section = menus.get(user_id, "main")
        if section == "knowledge":
            return knowledge_keyboard()
        if section == "memory":
            return memory_keyboard()
        return main_keyboard()

    @bot.message_handler(commands=["start"])
    def handle_start(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        pending.pop(uid, None)
        menus[uid] = "main"
        histories.pop(uid, None)
        name = message.from_user.first_name or "друг"
        bot.reply_to(
            message,
            f"Привет, {name}! Я RAG-помощник.\n\n{HELP_TEXT}",
            reply_markup=main_keyboard(),
        )

    @bot.message_handler(func=lambda message: message.text == BTN_CANCEL)
    def handle_cancel(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        pending.pop(uid, None)
        bot.reply_to(
            message,
            "Отменил. Можно задать вопрос или выбрать кнопку.",
            reply_markup=section_keyboard(uid),
        )

    @bot.message_handler(func=lambda message: message.text == BTN_BACK)
    def handle_back(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        pending.pop(uid, None)
        menus[uid] = "main"
        bot.reply_to(message, "Главное меню.", reply_markup=main_keyboard())

    @bot.message_handler(func=lambda message: message.text == BTN_HELP)
    def handle_help(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        pending.pop(uid, None)
        bot.reply_to(message, HELP_TEXT, reply_markup=section_keyboard(uid))

    @bot.message_handler(func=lambda message: message.text == BTN_CAT_KNOWLEDGE)
    def handle_knowledge_menu(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        pending.pop(uid, None)
        menus[uid] = "knowledge"
        bot.reply_to(
            message,
            "База знаний: поиск, заметки и статьи по URL.",
            reply_markup=knowledge_keyboard(),
        )

    @bot.message_handler(func=lambda message: message.text == BTN_CAT_MEMORY)
    def handle_memory_menu(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        pending.pop(uid, None)
        menus[uid] = "memory"
        bot.reply_to(
            message,
            "Память: факты о вас сохраняются отдельно от общей базы.",
            reply_markup=memory_keyboard(),
        )

    @bot.message_handler(func=lambda message: message.text == BTN_SEARCH)
    def handle_search_button(message: telebot.types.Message) -> None:
        pending[user_id_of(message)] = "search"
        menus[user_id_of(message)] = "knowledge"
        bot.reply_to(
            message,
            "Напишите запрос для поиска по векторной базе.",
            reply_markup=cancel_keyboard(),
        )

    @bot.message_handler(func=lambda message: message.text == BTN_ADD_TEXT)
    def handle_add_button(message: telebot.types.Message) -> None:
        pending[user_id_of(message)] = "add"
        menus[user_id_of(message)] = "knowledge"
        bot.reply_to(
            message,
            "Пришлите текст, который нужно сохранить в общую базу знаний.",
            reply_markup=cancel_keyboard(),
        )

    @bot.message_handler(func=lambda message: message.text == BTN_ADD_URL)
    def handle_url_button(message: telebot.types.Message) -> None:
        pending[user_id_of(message)] = "url"
        menus[user_id_of(message)] = "knowledge"
        bot.reply_to(
            message,
            "Пришлите ссылку на HTML-статью. Я прочитаю страницу, разобью на чанки и запишу в базу.",
            reply_markup=cancel_keyboard(),
        )

    @bot.message_handler(func=lambda message: message.text == BTN_REMEMBER)
    def handle_remember_button(message: telebot.types.Message) -> None:
        pending[user_id_of(message)] = "remember"
        menus[user_id_of(message)] = "memory"
        bot.reply_to(
            message,
            "Напишите факт о себе, например: меня зовут Иван, живу в Казани.",
            reply_markup=cancel_keyboard(),
        )

    @bot.message_handler(
        content_types=["text"],
        func=lambda message: pending.get(message.from_user.id) == "search",
    )
    def handle_search_text(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        query = (message.text or "").strip()
        pending.pop(uid, None)
        if not query:
            bot.reply_to(message, "Пустой запрос не ищу.", reply_markup=knowledge_keyboard())
            return
        bot.send_chat_action(message.chat.id, "typing")
        try:
            docs = agent.search_knowledge(query, k=5)
            memory_docs = agent.search_user_memory(query, str(uid), k=3)
            if not docs:
                docs = agent.search(query, k=5)
            docs = memory_docs + docs
            answer = format_search_results(docs)
        except Exception as exc:
            answer = f"Не удалось выполнить поиск: {exc}"
        send_long(
            bot,
            message.chat.id,
            answer,
            reply_to=message.message_id,
            reply_markup=knowledge_keyboard(),
        )

    @bot.message_handler(
        content_types=["text"],
        func=lambda message: pending.get(message.from_user.id) == "add",
    )
    def handle_add_text(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        note = (message.text or "").strip()
        pending.pop(uid, None)
        if not note:
            bot.reply_to(message, "Пустой текст не сохранил.", reply_markup=knowledge_keyboard())
            return
        try:
            count = agent.add_knowledge(note, source=f"telegram:{uid}")
            bot.reply_to(
                message,
                f"Сохранил в базу знаний. Чанков: {count}.",
                reply_markup=knowledge_keyboard(),
            )
        except Exception as exc:
            bot.reply_to(message, f"Не удалось сохранить: {exc}", reply_markup=knowledge_keyboard())

    @bot.message_handler(
        content_types=["text"],
        func=lambda message: pending.get(message.from_user.id) == "url",
    )
    def handle_url_text(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        text = (message.text or "").strip()
        pending.pop(uid, None)
        urls = extract_urls(text)
        if not urls:
            bot.reply_to(
                message,
                "Не нашёл http/https ссылку. Пришлите URL статьи.",
                reply_markup=knowledge_keyboard(),
            )
            return
        bot.send_chat_action(message.chat.id, "typing")
        lines: list[str] = []
        for url in urls:
            try:
                count = agent.ingest_url(url)
                lines.append(f"{url}\nПрочитал и сохранил. Чанков: {count}.")
            except Exception as exc:
                lines.append(f"{url}\nНе удалось добавить: {exc}")
        send_long(
            bot,
            message.chat.id,
            "\n\n".join(lines),
            reply_to=message.message_id,
            reply_markup=knowledge_keyboard(),
        )

    @bot.message_handler(
        content_types=["text"],
        func=lambda message: pending.get(message.from_user.id) == "remember",
    )
    def handle_remember_text(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        fact = (message.text or "").strip()
        pending.pop(uid, None)
        if not fact:
            bot.reply_to(message, "Пустой факт не сохранил.", reply_markup=memory_keyboard())
            return
        try:
            agent.save_user_fact(fact, str(uid))
            bot.reply_to(message, "Запомнил это о вас.", reply_markup=memory_keyboard())
        except Exception as exc:
            bot.reply_to(message, f"Не удалось запомнить: {exc}", reply_markup=memory_keyboard())

    @bot.message_handler(
        content_types=["text"],
        func=lambda message: bool(message.text) and not message.text.startswith("/"),
    )
    def handle_text(message: telebot.types.Message) -> None:
        uid = user_id_of(message)
        question = (message.text or "").strip()
        if not question:
            return
        bot.send_chat_action(message.chat.id, "typing")
        try:
            answer = agent.ask(
                question,
                user_id=str(uid),
                history=list(histories[uid]),
            )
        except Exception as exc:
            answer = f"Не получилось ответить: {exc}"
        histories[uid].append({"role": "user", "content": question})
        histories[uid].append({"role": "assistant", "content": answer})
        send_long(
            bot,
            message.chat.id,
            answer,
            reply_to=message.message_id,
            reply_markup=section_keyboard(uid),
        )

    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    run_bot()
