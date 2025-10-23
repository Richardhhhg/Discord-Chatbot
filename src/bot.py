import discord
from discord.ext import commands

from llm import LLM

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True 

bot = commands.Bot(command_prefix="!", intents=intents)
llm = LLM(debug=True)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

async def generate_message_history(channel):
    history = []
    async for message in channel.history():
        if message.author == bot.user:
            history.append({"role": "assistant", "content": message.content})
        else:
            history.append({"role": "user", "content": f"[{message.author}] {message.content}"})
    history.reverse()
    return history

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if isinstance(message.channel, discord.DMChannel):
        history = await generate_message_history(message.channel)
        await llm.load_memory(history)
        print(history)
        response = await llm.generate(prompt=message.content, speaker=str(message.author.name))
        await message.channel.send(response)

if __name__ == "__main__":
    bot.run("TOKEN")
