# Dataset Description

This document provides details about the CSV files in the Himalayan Climbing Expeditions dataset.

## 1. exped.csv or expeditions.csv
- **Total Columns:** 65
- **Total Rows:** 11,425
- **Column Names:** 
  - expid, peakid, year, season, host, route1, route2, route3, route4, nation, leaders, sponsor
  - success1, success2, success3, success4, ascent1, ascent2, ascent3, ascent4
  - claimed, disputed, countries, approach, bcdate, smtdate, smttime, smtdays
  - totdays, termdate, termreason, termnote, highpoint, traverse, ski, parapente
  - camps, rope, totmembers, smtmembers, mdeaths, tothired, smthired, hdeaths
  - nohired, o2used, o2none, o2climb, o2descent, o2sleep, o2medical, o2taken
  - o2unkwn, othersmts, campsites, accidents, achievment, agency, comrte, stdrte
  - primrte, primmem, primref, primid, chksum

## 2. himalayan_data_dictionary.csv
- **Total Columns:** 3
- **Total Rows:** 165
- **Column Names:**
  - Table
  - Field
  - Description

## 3. members.csv
- **Total Columns:** 61
- **Total Rows:** 89,000
- **Column Names:**
  - expid, membid, peakid, myear, mseason, fname, lname, sex, yob, citizen
  - status, residence, occupation, leader, deputy, bconly, nottobc, support
  - disabled, hired, sherpa, tibetan, msuccess, mclaimed, mdisputed, msolo
  - mtraverse, mski, mparapente, mspeed, mhighpt, mperhighpt, msmtdate1
  - msmtdate2, msmtdate3, msmttime1, msmttime2, msmttime3, mroute1, mroute2
  - mroute3, mascent1, mascent2, mascent3, mo2used, mo2none, mo2climb
  - mo2descent, mo2sleep, mo2medical, mo2note, death, deathdate, deathtime
  - deathtype, deathhgtm, deathclass, msmtbid, msmtterm, hcn, mchksum

## 4. peaks.csv
- **Total Columns:** 23
- **Total Rows:** 480
- **Column Names:**
  - peakid, pkname, pkname2, location, heightm, heightf, himal, region
  - open, unlisted, trekking, trekyear, restrict, phost, pstatus, pyear
  - pseason, pmonth, pday, pexpid, pcountry, psummiters, psmtnote

## 5. refer.csv
- **Total Columns:** 12
- **Total Rows:** 15,586
- **Column Names:**
  - expid, refid, ryear, rtype, rjrnl, rauthor
  - rtitle, rpublisher, rpubdate, rlanguage, rcitation, ryak94

