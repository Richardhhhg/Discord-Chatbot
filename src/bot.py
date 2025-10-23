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

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Check if message is in a DM
    if isinstance(message.channel, discord.DMChannel):
        print(f"Received DM from {message.author}: {message.content}")
        
        response = await llm.generate(prompt=message.content, speaker=str(message.author.name))
        await message.channel.send(response)

    # Process commands normally in servers
    await bot.process_commands(message)

bot.run("TOKEN HERE")
